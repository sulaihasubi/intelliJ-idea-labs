package global.skymind.solution.modelsaveload.TESTmyself;


import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.ILoggerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**

 *  This examples builds on the MnistImagePipelineExample
 *  by loading the trained network
 *
 * To run this sample, you must have
 * (1) save trained mnist model
 * (2) test image
 *
 *  Look for LAB STEP below. Uncomment to proceed.
 *  1. Load the saved model
 *  2. Load an image for testing
 *  3. [Optional] Preprocessing to 0-1 or 0-255
 *  4. Pass through the neural net for prediction
 */

public class testmyself_MnistImageLoad {

    private static Logger log = LoggerFactory.getLogger(testmyself_MnistImageLoad.class);

    public static void main(String[] args) throws Exception
    {
        //image information
        //28 by 28 grayscale image
        //grayscale implies single channel

        int height = 28;
        int width = 28;
        int channels = 1; //single channel

        File modelSave =  new File(System.getProperty("java.io.tmpdir"), "/trained_mnist_model.zip"); // loads images from disk

        //loop function to check the condition we
        if(modelSave.exists() == false ){
            System.out.println("Directory Not Found! System Abort.");
            return;
        }
        File imageToTest =  new ClassPathResource("image/five.png").getFile();

        //Step 1: Load the save model from the neural network
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);

        //Step 2: Load image for testing
        //Use NativeImageLoader to convert to numerical matrix
        NativeImageLoader loader = new NativeImageLoader(height,width,channels);

        //get the image into INDArray. INDArray is N-dimensional array or tensors
        INDArray image = loader.asMatrix(imageToTest);

        //Step 3: Data Normalization (optional)
        //preprocessing to 0-1 or 0-255
        DataNormalization scaler = new ImagePreProcessingScaler(0,1); //ImagePreProcessingScaler is obviously a good choice for image data.
        scaler.transform(image);

        //Step 4 Pass to the neural network for prediction
        INDArray output = model.output(image);
        log.info("The Label is:                   " + Nd4j.argMax(output,1));
        log.info("The Probabilities is :                " + output.toString());
    }

}
