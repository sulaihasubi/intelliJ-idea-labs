package global.skymind.solution.VAE;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class CatVsDogClassification {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(CatVsDogClassification.class);

    private static final int width = 224;
    private static final int height = 224;
    private static final int channel = 3;
    private static final int batchSize = 10;
    private static final int numOfClass = 2;
    private static final int epochs = 50;
//    numOfImage=25000, batchSize is 10, how many iteration it takes to complete the epoch? 25000/10=2500, 25000/50=
    public static void main(String[] args) throws IOException {
        String homePath = System.getProperty("user.home");

        Path dataPath = Paths.get(homePath, "dataset","train");
        System.out.println(dataPath.toString());

        int seed = 1234;
        Random randNumGen = new Random(seed);
        String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        FileSplit fileSplit = new FileSplit(new File(dataPath.toString()));

        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelGenerator);

        InputSplit[] trainTestSplit = fileSplit.sample(balancedPathFilter, 80,20);
        InputSplit trainData = trainTestSplit[0];
        InputSplit testData = trainTestSplit[1];

        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width,channel,  labelGenerator);
        ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channel, labelGenerator);

        trainRecordReader.initialize(trainData);
        testRecordReader.initialize(testData);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, numOfClass);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, numOfClass);

        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);


        ZooModel zooModel = ResNet50.builder().build();
        ComputationGraph resnet = (ComputationGraph) zooModel.initPretrained();
        log.info(resnet.summary());

//        // Override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(1e-3))
                .seed(seed)
                .build();
//
        ComputationGraph resnet50Transfer = new TransferLearning.GraphBuilder(resnet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("bn5b_branch2c") //"bn5b_branch2c" and below are frozen
                .addLayer("fc",new DenseLayer
                        .Builder().activation(Activation.RELU).nIn(1000).nOut(256).build(),"fc1000") //add in a new dense layer
                .addLayer("newpredictions",new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(256)
                        .nOut(numOfClass)
                        .build(),"fc")
                .setOutputs("newpredictions")
                .build();

        log.info(resnet50Transfer.summary());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);

        resnet50Transfer.setListeners(
                new StatsListener(statsStorage),
                new ScoreIterationListener(50)
        );

        resnet50Transfer.fit(trainIter, epochs);

        Evaluation trainEval = resnet50Transfer.evaluate(trainIter);
        Evaluation testEval = resnet50Transfer.evaluate(testIter);
        System.out.println(trainEval.stats());
        System.out.println(testEval.stats());

        ModelSerializer.writeModel(resnet50Transfer, homePath+"/model/catvsdog_resnet50.zip", false);
    }
}
