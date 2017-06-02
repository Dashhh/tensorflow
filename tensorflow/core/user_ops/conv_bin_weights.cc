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

REGISTER_OP("BinaryConvWeight2D")
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
            filter_abs_sum += filter_source_value > 0 ? filter_source_value : -filter_source_value;
          }
        }
      }
    }
    
    float alpha = filter_abs_sum / entries_counter;        

    for (batch = 0; batch < input_batches; ++batch) {
      int** D = new int*[output_height];
      int** input_compressed = new int*[output_height];
      for(int i = 0; i < output_height; ++i) {
        D[i] = new int[output_width];
        input_compressed = new int*[output_width];
      }
      //K calculation
      float* K = new float[output_height * output_width];
      for (out_y = 0; out_y < output_height; ++out_y) {
        for (out_x = 0; out_x < output_width; ++out_x) {
          float compressed = 0.0;
          const int in_y = out_y;
          const int in_x = out_x; 
          for (out_channel = 0; out_channel < input_depth; ++out_channel) {
            compressed += input_data[(batch * input_height * input_width * input_depth) + (in_y * input_width * input_depth) + (in_x * input_depth) + out_channel;
          }
          input_compressed[out_y][out_x] = compressed / input_depth;
        }
      }
      // powieksz o rozmiar filtra
      D[0][0] = input_compressed[0][0];
      for (int i = 1; i < output_height; i++) {
        D[0][i] = D[0][i-1] + input_compressed[0][i];
        D[i][0] = D[i-1][0] + input_compressed[i][0];
      }
      for (int i = 1; i < output_height; i++) {
        for (int j = 1; j < output_weight; j++) {
          D[i][j] = D[i-1][j] + D[i][j-1] - D[i-1][j-1] + input_compressed[i][j];
        }
      }
      for (out_y = 1; out_y < output_height; ++out_y) {
        for (out_x = 1; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride) - filter_left_offset;
          const int in_y_origin = (out_y * stride) - filter_top_offset;
          float total = 0;
          if (out_y > filter_hegith
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
                  const T1 input_value = input_source_value > 0 ? input_source_value : -input_source_value;
                  const T2 filter_source_value =
                     1. / (float) (filter_height * filter_width * filter_count);
                  //sign function
                  //const int input_value = static_cast<int>((input_source_value > 0) - (input_source_value < 0));
                  total += (input_value * filter_source_value);
                }
              }
            }
            K[out_y*output_width + out_x] = total;
          }
        }
      }
    
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
                  //sign function
                  const int filter_value = static_cast<int>((filter_source_value > 0) - (filter_source_value < 0));
                  const int input_value = static_cast<int>((input_source_value > 0) - (input_source_value < 0));
                  total += (input_value * filter_value);
                }
              }
            }
            const float beta = K[out_y*output_width + out_x];
	    //std::cout<<"beta: "<<beta<<std::endl;
            const float output = total * alpha * beta;
            output_data[(batch * output_height * output_width * filter_count) +
                        (out_y * output_width * filter_count) +
                        (out_x * filter_count) + out_channel] = output;
          }
        }
      }
    delete K;
    }
  }
};

template <class T1, class T2, class T3,
          template <class TF1, class TF2, class TF3> class ConvFunctor>
class BinaryConv2DOp : public OpKernel {
 public:
  explicit BinaryConv2DOp(OpKernelConstruction* context)
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
    Name("BinaryConvWeight2D")
        .Device(DEVICE_CPU)
        .TypeConstraint<float>("T"),
    BinaryConv2DOp<float,float,float, ReferenceConvFunctor>);

}  // namespace tensorflow
