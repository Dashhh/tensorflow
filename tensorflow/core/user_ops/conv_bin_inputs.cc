#include <algorithm>
#include <vector>

#include <omp.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("BinaryConvInput2D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float32, float64}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
conv_bin2D.
)doc");

template <class T1, class T2, class T3>
class ReferenceConvFunctor {
 int flsign(float x){
   return x > 0 ? 1 : -1;
 }
 float flabs(float x){
   return x > 0 ? x : -x;
 }
 public:
  void operator()(OpKernelContext* context, const T1* input_data,
                  int input_batches, int input_height, int input_width,
                  int input_depth, const T2* filter_data,
                  int filter_height, int filter_width, int filter_count,
                  int stride, Padding padding,
                  T3* output_data, int output_height, int output_width) {
    int filter_left_offset;
    int filter_top_offset;
    if (padding == VALID) {
      filter_left_offset =
          ((output_width - 1) * stride + filter_width - input_width + 1) / 2;
      filter_top_offset =
          ((output_height - 1) * stride + filter_height - input_height + 1) / 2;
    } else {
      filter_left_offset =
          ((output_width - 1) * stride + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride + filter_height - input_height) / 2;
    }

    int entries_counter = filter_height * filter_width * input_depth;
    float filter_abs_sum = 0.0;

    int out_channel, filter_y, filter_x, in_channel, batch, out_y, out_x;

    //alpha calculation.
    #pragma omp parallel for collapse(4) private(out_channel, filter_y, filter_x, in_channel)
    for (out_channel = 0; out_channel < filter_count; ++out_channel) {
      for (filter_y = 0; filter_y < filter_height; ++filter_y) {
        for (filter_x = 0; filter_x < filter_width; ++filter_x) {
          for (in_channel = 0; in_channel < input_depth; ++in_channel) {
            const T2 filter_source_value =
              filter_data[(filter_y * filter_width * input_depth *
                          filter_count) + (filter_x * input_depth * filter_count) +
                          (in_channel * filter_count) + out_channel];
            filter_abs_sum += flabs(filter_source_value);
          }
        }
      }
    }

    float alpha = filter_abs_sum / entries_counter;
    for (batch = 0; batch < input_batches; ++batch) {

      //input_compressed initialization
      float** input_compressed = new float*[input_height];
      for(int i = 0; i < input_height; ++i) {
        input_compressed[i] = new float[input_width];
      }

      //D initialization
      int d_height = input_height + 1; //0 padding
      int d_width = input_width + 1; //0 padding
      float** D = new float*[d_height];
      for(int i=0;i<d_height;++i){
        D[i] = new float[d_width];
      }

      //K initialization
      int k_height = input_height - filter_height + 1;
      int k_width = input_width - filter_width + 1;
      float** K = new float*[k_height];
      for(int i=0;i<k_height;++i){
        K[i] = new float[k_width];
      }

      for (int in_y = 0; in_y < input_height; ++in_y) {
        for (int in_x = 0; in_x < input_width; ++in_x) {
          float compressed = 0.0;
          std::cout << in_x << in_y << std::endl;
          for (out_channel = 0; out_channel < input_depth; ++out_channel) {
            float input = input_data[(batch * input_height * input_width * input_depth) +
            (in_y * input_width * input_depth) + (in_x * input_depth) + out_channel];
            compressed += flabs(input);
          }
          std::cout<<"compressed "<<compressed<<std::endl;
          std::cout<<"input_depth "<<input_depth<<std::endl;
          input_compressed[in_y][in_x] = compressed / input_depth;
        }
      }

      std::cout << "input compressed[" << input_height << "][" << input_width << "]" << std::endl;
      std::cout << "K[" << k_height << "][" << k_width << "]" << std::endl;
      std::cout << "D[" << d_height << "][" << d_width << "]" << std::endl;

      //powieksz o rozmiar filtra
      D[1][1] = input_compressed[0][0];
      D[0][0] = 0;
      for (int i = 1; i < d_height; i++) {
        D[i][0] = 0;
        D[i][1] = D[i-1][1] + input_compressed[i-1][0];
      }
      for (int i = 1; i < d_width; i++){
        D[0][i] = 0;
        D[1][i] = D[1][i-1] + input_compressed[0][i-1];
      }
      std::cout<<"finished initializing borders of D\n";
      for (int i = 2; i < d_width; i++) {
        for (int j = 2; j < d_height; j++) {
          D[i][j] = D[i-1][j] + D[i][j-1] - D[i-1][j-1] + input_compressed[i-1][j-1];
        }
      }
      for(int i=0;i<d_height;i++){
        for(int j=0;j<d_width;j++)
          {
              std::cout<<D[i][j]<<" ";
          }
          std::cout<<"\n";
      }

      for(int i=0;i<k_height;i++){
        for(int j=0;j<k_width;j++){
	        int h = filter_height;
          int w = filter_width;
          K[i][j] = D[i+h][j+w] - D[i+h][j] - D[i][j+w] + D[i][j];
          K[i][j] /= h*w;
          std::cout<<"K["<<i<<"]["<<j<<"]=" << K[i][j] << '\n';
        }
      }

      //todo: xnor bitcount
      #pragma omp parallel for collapse(3) private(out_channel, out_y, out_x)
      for (out_y = 0; out_y < output_height; ++out_y) {
        for (out_x = 0; out_x < output_width; ++out_x) {
          for (out_channel = 0; out_channel < filter_count; ++out_channel) {
            const int in_x_origin = (out_x * stride) - filter_left_offset;
            const int in_y_origin = (out_y * stride) - filter_top_offset;
            float total = 0;
            //float input_abs_sum = 0.0;
           for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  T1 input_source_value;
                  if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height)) {
                    input_source_value =
                        input_data[(batch * input_height * input_width *
                                    input_depth) +
                                   (in_y * input_width * input_depth) +
                                   (in_x * input_depth) + in_channel];

                  }
                  else {
                    input_source_value = 0;
                  }
                  //input_abs_sum += abs(input_source_value);
                  const T2 filter_source_value =
                      filter_data[(filter_y * filter_width * input_depth *
                                   filter_count) +
                                  (filter_x * input_depth * filter_count) +
                                  (in_channel * filter_count) + out_channel];

                  const int filter_value = flsign(filter_source_value);
                  const int input_value = flsign(input_source_value);
                  total += (input_value * filter_value);
                }
              }
            }
            const float beta = K[out_y][out_x];
            const float output = total * alpha * beta;
            output_data[(batch * output_height * output_width * filter_count) +
                        (out_y * output_width * filter_count) +
                        (out_x * filter_count) + out_channel] = output;
          }
        }
      }
    //todo: nie tylko delete K ale też chyba tych pod spodem wierszy
    //delete K;
    //delete input_compressed;
    //delete D;
    }
  }
};

template <class T1, class T2, class T3,
          template <class TF1, class TF2, class TF3> class ConvFunctor>
class BinaryConvInput2DOp : public OpKernel {
 public:
  explicit BinaryConvInput2DOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_[1] == strides_[2],
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
     OP_REQUIRES(
    	    context, (strides_[0] == 1 && strides_[3] == 1),
       errors::InvalidArgument("Current implementation does not yet support "
                               "strides in the batch and depth dimensions."));
   OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    const Tensor& filter = context->input(1);

    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    const int64 in_depth = input.dim_size(3);
    OP_REQUIRES(context, in_depth == filter.dim_size(2),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", in_depth,
                    " vs ", filter.dim_size(2)));

    const int64 out_depth = filter.dim_size(3);

    const int64 input_rows = input.dim_size(1);
    const int64 filter_rows = filter.dim_size(0);

    const int64 input_cols = input.dim_size(2);
    const int64 filter_cols = filter.dim_size(1);

    const int64 batch = input.dim_size(0);

    const int stride = strides_[1];

    int64 out_rows = 0;
    int64 out_cols = 0;
    int64 pad_rows = 0;
    int64 pad_cols = 0;

    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride,
                                         padding_, &out_cols, &pad_cols));

    CHECK_GT(batch, 0);
    CHECK_GT(out_rows, 0);
    CHECK_GT(out_cols, 0);
    CHECK_GT(out_depth, 0);
    TensorShape out_shape({batch, out_rows, out_cols, out_depth});

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    ConvFunctor<T1, T2, T3> conv_functor;
    conv_functor(context, input.flat<T1>().data(), batch, input_rows,
                 input_cols, in_depth, filter.flat<T2>().data(),
                 filter_rows, filter_cols, out_depth, stride,
                 padding_, output->flat<T3>().data(), out_rows, out_cols);

  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
};

REGISTER_KERNEL_BUILDER(
    Name("BinaryConvInput2D")
        .Device(DEVICE_CPU)
        .TypeConstraint<float>("T"),
    BinaryConvInput2DOp<float,float,float, ReferenceConvFunctor>);

}  // namespace tensorflow
