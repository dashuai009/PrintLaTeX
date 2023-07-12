#include "../opencv_image_io.hpp"

int main(int argc, const char *argv[]) {
    std::string picName{"1f4a90ef62"};
    std::string msg = "sample image ";
    msg += picName;
    auto currentPath =
            std::string("C:\\Users\\15258\\work\\PrintLaTeX\\main\\data\\formula_images\\") + picName + ".png";
    auto img_tensor = image_io::ReadImage_gray(currentPath);
    auto img_t = image_io::ToCvImage(img_tensor, CV_8UC1);
    image_io::test::show_image(img_t, msg);


    auto img = cv::imread(currentPath, cv::IMREAD_GRAYSCALE);
    image_io::test::show_image(img, msg);
//    // convert the cvimage into tensor
//    auto tensor = ToTensor(img);
//
//    // preprocess the image. meaning alter it in a way a bit!
//    tensor = tensor.clamp_max(c10::Scalar(50));
//
//    auto cv_img = ToCvImage(tensor);
//    show_image(cv_img, "converted image from tensor");
//    // convert the tensor into float and scale it
//    tensor = tensor.toType(c10::kFloat).div(255);
//    // swap axis
//    tensor = transpose(tensor, {(2), (0), (1)});
//    //add batch dim (an inplace operation just like in pytorch)
//    tensor.unsqueeze_(0);
//
//    auto input_to_net = ToInput(tensor);
//
//
//    torch::jit::script::Module r18;
//
//    try {
//        std::string r18_model_path = "D:\\Codes\\python\\Model_Zoo\\jitcache\\resnet18.pt";
//
//
//        // Deserialize the ScriptModule from a file using torch::jit::load().
//        r18 = torch::jit::load(r18_model_path);
//
//        // Execute the model and turn its output into a tensor.
//        at::Tensor output = r18.forward(input_to_net).toTensor();
//
//        //sizes() gives shape.
//        std::cout << output.sizes() << std::endl;
//        std::cout << "output: " << output[0] << std::endl;
//        //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
//
//    }
//    catch (const c10::Error &e) {
//        std::cerr << "error loading the model\n" << e.msg();
//        return -1;
//    }
//
//    std::cout << "ok\n";
//    std::system("pause");
    return 0;
}
