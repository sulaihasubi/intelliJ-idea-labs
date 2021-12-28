package global.skymind.solution.convolution.objectdetection;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.YOLO2;
import org.deeplearning4j.zoo.util.darknet.COCOLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class test {
    private static final int gridWidth = 13;
    private static final int gridHeight = 13;
    private static double detectionThreshold = 0.5;
    private static final int yolowidth = 416; //input image size
    private static final int yoloheight = 416; //input image size


    public static void main(String[] args) throws Exception {

        String testImagePATH = "C:\\Users\\user\\Desktop\\SkyMind Training\\Screenshot\\Kinabalu.jpg";
//        String testImagePATH = "C:\\Users\\choowilson\\Pictures\\events\\test.jpg";

        File file = new File(testImagePATH);
        System.out.println(String.format("You are using this image file located at %s", testImagePATH));
        COCOLabels labels = new COCOLabels();
        System.out.println(labels);

        ZooModel yolo2 = YOLO2.builder().numClasses(0).build();
        ComputationGraph model = (ComputationGraph) yolo2.initPretrained();

        NativeImageLoader nil = new NativeImageLoader(yolowidth, yoloheight, 3);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        INDArray image = nil.asMatrix(file);
        scaler.transform(image);




//        Actually this part is for the visualisation purposed. We dont need that, just make it comment.
//        System.out.println(Arrays.toString(image.shape()));
//        int w = (int) image.shape()[2];
//        int h = (int) image.shape()[3];

        Mat opencvMat = imread(testImagePATH);
//        resize original image size to the target size for prediction (416*416)
//        resize(opencvMat, opencvMat, new Size(w, h));
        int w = opencvMat.cols();
        int h = opencvMat.rows();

        INDArray outputs = model.outputSingle(image);
        System.out.println(Arrays.toString(outputs.shape()));
        List<DetectedObject> objs = YoloUtils.getPredictedObjects(Nd4j.create(((YOLO2) yolo2).getPriorBoxes()), outputs, detectionThreshold, 0.4); //helps to inteprate the output. Detector object

        for (DetectedObject obj : objs) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            String label = labels.getLabel(obj.getPredictedClass());
            int x1 = (int) Math.round(w * xy1[0] / gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / gridHeight);
            rectangle(opencvMat, new Point(x1, y1), new Point(x2, y2), Scalar.RED, 2, 0, 0);
            putText(opencvMat, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN); //captioning for the image
        }
//        resize to see the input and prediction more clearly
//        resize(opencvMat, opencvMat, new Size(w * 2, h * 2));
        imshow("Input Image", opencvMat);

//        Press "Esc" to close window
        if (waitKey(0) == 27) {
            destroyAllWindows();
        }
    }
}
