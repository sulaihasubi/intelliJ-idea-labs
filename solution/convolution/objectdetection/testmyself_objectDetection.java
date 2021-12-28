//package global.skymind.solution.convolution.objectdetection;
//
//import org.bytedeco.opencv.opencv_core.Mat;
//import org.bytedeco.opencv.opencv_core.Point;
//import org.bytedeco.opencv.opencv_core.Scalar;
//import org.bytedeco.opencv.opencv_core.Size;
//import org.datavec.image.loader.NativeImageLoader;
//import org.deeplearning4j.nn.graph.ComputationGraph;
//import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
//import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
//import org.deeplearning4j.zoo.ZooModel;
//import org.deeplearning4j.zoo.model.YOLO2;
//import org.deeplearning4j.zoo.util.darknet.COCOLabels;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
//import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
//import org.nd4j.linalg.factory.Nd4j;
//
//import java.io.File;
//import java.util.Arrays;
//import java.util.List;
//
//import static org.bytedeco.opencv.global.opencv_highgui.*;
//import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
//import static org.bytedeco.opencv.global.opencv_imgproc.*;
//
//// in this code i will test my self to load an image from the file and do the object detection in that image
//
//public class testmyself_objectDetection {
//
//    // image size must be in 13 x 13 size
//    //yolo input image musbe in 416 x 416 size
//    public static final int gridWidth = 13; //width of the image 13
//    public static final int gridHeight = 13; // height of the image is 13
//    private static double detectThreshold = 0.5; //0.5 is default value
//    public static final int yolowidth = 416; // input image width
//    public static final int yoloheight = 416; // input image height
//
//
//    //pembuka bicara of the code
//    //dalam psvm akan dimulakan segala process
//    public static void main(String[] args) throws Exception {
//
//        // Step 1: Load image from the path. Image source is from Google
//        String testImagePath = "C:\\Users\\user\\Desktop\\car.jpg";
//
//        File file = new File(testImagePath);
//        System.out.println(String.format("You are using the Image that located in %s", testImagePath)); //call out the directory of the
//        // image save and located
//        COCOLabels labels = new COCOLabels();
//        System.out.println(labels);
//
//        //Step 2: Use ZooModel the image recognition configurations
//        ZooModel yolo2 = YOLO2.YOLO2Builder().numClasses(0).build();
//        ComputationGraph model = (ComputationGraph) yolo2.initPretrained();
//
//        NativeImageLoader nil = new NativeImageLoader(yoloheight, yolowidth, 3);
//        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1); //A preprocessor specifically for images that applies min max scaling
//
//        //transform image to matrix
//        INDArray image = nil.asMatrix(file);
//        scaler.transform(image);
//
//
//    };
//
//
//
//
//
//
//
//
//    }
//
//
//
//};
