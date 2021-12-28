package global.skymind.solution.modelsaveload.TESTmyself;

import global.skymind.solution.modelsaveload.MnistImageSave;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

//in this code i will try to download the data set images and then build Mnist Example Image Pipeline by saving the Trained Network!
//MNIST data set iterator - 60000 training digits, 10000 test digits, 10 classes. Digits have 28x28 pixels and 1 channel (grayscale).

public class testmyself_MnistImageSave {
    /** Data URL for downloading */
    private static Logger log = LoggerFactory.getLogger(MnistImageSave.class);

    public static void main(String[] args) throws Exception
    {

        //image information
        //28 by 28 grayscale image
        //grayscale implies single channel

        int height = 28;
        int width = 28;
        int channels = 1;
        int rngseed = 123; //RNG Seed: If your model uses probabilities (i.e. DropOut/DropConnect),
        // it may make sense to save it separately, and apply it after model is restored


        Random randNumGen = new Random(rngseed);
        int BatchSize = 130; //IteratorDataSetIterator, so it is required to get the specified batch size (batch size for each epoch)
        int OutputNum = 10; //number of output classes
        int NumofEpoch = 1;

        //Dataset Iterator
        //Pass the MNIST data iterator that automatically fetches data
        DataSetIterator MnistTrain = new MnistDataSetIterator(BatchSize,true,rngseed);

        //Scale pixel value to 0 and 1
        //Before training the neural network, we will instantiate built-in DataSetIterators for the MNIST data
        //One example of data preprocessing is scaling the data. The data we are using in raw form are greyscale images,
        // which are represented by a single matrix filled with integer values from 0 to 255.
        // 0 value indicates a black pixel, while a 1 value indicates a white pixel.
        // It is helpful to scale the image pixel value from 0 to 1 instead of from 0 to 255.
        //To do this, the ImagePreProcessingScaler class is used directly on the MnistDataSetIterators.
        //Note that this process is typtical for data preprocessing. Once this is done, we are ready to train the neural network

        DataNormalization Scaler = new ImagePreProcessingScaler(0,1);
        Scaler.fit(MnistTrain);
        MnistTrain.setPreProcessor(Scaler);

        //Start to build the Neural Network
        log.info("**** Build Model ****");

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngseed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs())
            .l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder()
                    .nIn(height * width)
                    .nOut(100)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(100)
                    .nOut(OutputNum)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .backpropType(BackpropType.Standard)
            .setInputType(InputType.convolutional(height,width,channels))
            .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        log.info("*****TRAIN MODEL********");
        for(int i = 0; i<NumofEpoch; i++)
    {
        model.fit(MnistTrain);//train the network by simply calling the fit method (to begin a task for training)
    }


        log.info("******SAVE TRAINED MODEL******");
        // Where to save the model
        File locationToSave = new File(System.getProperty("java.io.tmpdir"), "/trained_mnist_model.zip");
        log.info(locationToSave.toString());

        // boolean save Updater
        boolean saveUpdater = false;

        // ModelSerializer needs modelname, saveUpdater, Location
        ModelSerializer.writeModel(model,locationToSave,saveUpdater); //this is for save purpose by simply calling the modelserializer method

        log.info("******PROGRAM IS FINISHED PLEASE CLOSE******");

}
}
