#include <iostream>
#include <cstdio>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

using namespace cv;
using namespace tflite;
using namespace std;
#define LOG(x) std::cerr
#define TFLITE_MINIMAL_CHECK(x)                                  \
	if (!(x))                                                    \
	{                                                            \
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
		exit(1);                                                 \
	}

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		fprintf(stderr, "minimal <tflite model>\n");
		return 1;
	}
	const char *filename = argv[1];

	// Load model
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
	TFLITE_MINIMAL_CHECK(model != nullptr);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	InterpreterBuilder builder(*model, resolver);
	std::unique_ptr<Interpreter> interpreter;
	builder(&interpreter);
	TFLITE_MINIMAL_CHECK(interpreter != nullptr);

	// Allocate tensor buffers.
	TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
	printf("=== Pre-invoke Interpreter State ===\n");
	// tflite::PrintInterpreterState(interpreter.get());

	// get input dimension from the input tensor metadata
	// assuming one input only
	int input = interpreter->inputs()[0];
    if (1)
        LOG(INFO) << "input: " << input << "\n";

    const std::vector<int> inputs = interpreter->inputs();
    const std::vector<int> outputs = interpreter->outputs();

    if (1)
    {
        LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
        LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
    }
	TfLiteIntArray *dims = interpreter->tensor(input)->dims;
	int wanted_height = dims->data[1];
	int wanted_width = dims->data[2];
	int wanted_channels = dims->data[3];

	LOG(INFO) << "wanted_height：" << wanted_height << "\n"<< "wanted_width:"<<wanted_width<< "\n"<< "wanted_channels:"<< wanted_channels << "\n";
	LOG(INFO) << "输入类型："<< interpreter->tensor(input)->type << endl;

	//OpenCV加载图片

    cv::Mat img = cv::imread("./img/user_6.jpg");
    int cols = wanted_width;
    int rows = wanted_height;
    cv::Size dsize = cv::Size(cols, rows);
    cv::Mat img_rs;
    cv::resize(img, img_rs, dsize);
    cv::cvtColor(img_rs, img_rs, cv::COLOR_BGR2RGB);

	auto img_inputs = interpreter->typed_tensor<float>(input);
	for (int i = 0; i < img_rs.cols * img_rs.rows * 3; i++)
	{
		img_inputs[i] = (float)img_rs.data[i];
	}
	// Fill input buffers
	// TODO(user): Insert code to fill input tensors

	// Run inference
	TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
	printf("\n\n=== Post-invoke Interpreter State ===\n");
	// tflite::PrintInterpreterState(interpreter.get());

	// Read output buffers
	// TODO(user): Insert getting data out code.
	int output = interpreter->outputs()[0];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
	auto output_size = output_dims->data[output_dims->size - 1];
	LOG(INFO) << "output_size = "<< output_size <<endl;
	cout << "Output dims" << endl;
	for (int i = 0; i < 4; i++)
	{
		cout << output_dims->data[i] << endl;
	}
	auto* score_map = interpreter->typed_output_tensor<float>(0);
	for (size_t i = 0; i < output_size; i++)
	{
		std::printf("%ld\t: %6.2f%%\n", i, score_map[i] * 100.f);
	}
	return 0;
}
