package global.skymind.solution.homework;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;


public class OilPalmClassification {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(OilPalmClassification.class);

    public static void main(String[] args) throws Exception{
        String homePath = System.getProperty("user.home");

        Path dataPath = Paths.get(homePath, "dataset","OilPalm_Images");
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

        ImageRecordReader trainRecordReader = new ImageRecordReader(224, 224,3,  labelGenerator);
        ImageRecordReader testRecordReader = new ImageRecordReader(224, 224, 3, labelGenerator);

        trainRecordReader.initialize(trainData);
        testRecordReader.initialize(testData);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, 32, 1, 2);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader, 32, 1, 2);

        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .l2(1e-4)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{11,11}, new int[]{4, 4})
                        .name("cnn1")
                        .nIn(3)
                        .nOut(96)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .name("maxpool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, new int[]{2,2})
                        .name("cnn2")
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3}, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(500)
                        .activation(Activation.RELU)
                        .dropOut(0.2)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(500)
                        .dropOut(0.2)
                        .build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(224, 224, 3))
                .build();

        //train model and eval model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info(model.summary());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);

        model.setListeners(
                new StatsListener(statsStorage, 5),
                new ScoreIterationListener(5)
        );

//        for(int i=0;i< 10;i++){
//            if(trainIter.hasNext()==true)
//            {
//                model.fit(trainIter.next());
//            }
//            Evaluation eval = model.evaluate(testIter);
//            log.info(eval.stats());
//        }

        model.fit(trainIter, 5);

        Evaluation eval = model.evaluate(testIter);
        log.info(eval.stats());
    }


}
