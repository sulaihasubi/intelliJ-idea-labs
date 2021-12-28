/*
 *
 *  * ******************************************************************************
 *  *  * Copyright (c) 2019 Skymind AI Bhd.
 *  *  * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package global.skymind.solution.feedforward.detectgender;


import org.datavec.api.split.FileSplit;
import org.deeplearning4j.arbiter.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * - Notes:
 *  - Data files are stored at following location
 *  - each character represented by 5 binary digit
 *  - so input nodes = maxlengthname * 5 (shorter name will have padding to the right)
 *  resources\PredictGender\Data folder
 */

public class PredictGenderTrain
{
    public String filePath;


    public static void main(String args[]) throws Exception
    {
        PredictGenderTrain dg = new PredictGenderTrain();
        dg.filePath = new ClassPathResource("PredictGender").getFile().getAbsolutePath() + "/Data/";

        dg.train();
    }

    /**
     * This function uses GenderRecordReader and passes it to RecordReaderDataSetIterator for further training.
     */
    public void train() throws Exception
    {
        int seed = 123456;
        double learningRate = 0.08;
        int batchSize = 100;
        int nEpochs = 300;
        int numInputs = 0; //set later
        int numOutputs = 0;

        List<String> outputLabel = Arrays.asList(new String[]{"M", "F"}); //hey! i want to predict female or male.

        GenderRecordReader rrTrain = new GenderRecordReader(outputLabel); // i need record reader gender (this is customized, the advanced one)

        FileSplit fileSplit = new FileSplit(new File(this.filePath), new String[]{"csv"}); // where my files located and shows the path
        rrTrain.initialize(fileSplit); // initialize it by record reader using file splits

        // start creating my neural networks
        numInputs = rrTrain.getNameMaxLength() * 5;  // multiplied by 5 as for each letter we use five binary digits like 00000 (this 5 digits represent the character)
        // So i find the length of name and then times with 5

        numOutputs = 2; //number of output labels (males and females)
        //numHiddenNodes = 2 * numInputs + numOutputs; // this is arbiter (can set any number to find the best)

        //numInputs value is the label position
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rrTrain, batchSize, numInputs, 2);

        //creating the neural network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            //.biasInit(1)
            //.l2(1e-4)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //algorithm
            .updater(new Nesterovs(learningRate, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))//updater(new Adam())
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(numInputs)
                .nOut(500) //500 hidden nodes
                .activation(Activation.TANH)
                .build())
            .layer(1, new DenseLayer.Builder()
                .nIn(500) //must connected with the previous value
                .nOut(250)
                .activation(Activation.TANH)
                .build())
            .layer(2, new DenseLayer.Builder()
                .nIn(250) //also connected with the previous value
                .nOut(100)
                .activation(Activation.TANH)
                .build())
            .layer(3, new OutputLayer.Builder()
                .nIn(100)
                .nOut(numOutputs)
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .build())
            .backpropType(BackpropType.Standard) //this is optional. folder version
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf); //start building the model
        model.init();

        //Set UIServer
        /*
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        */
        //copy this every time you code to produce localhost 9000 that shows the graph (set UIServer)


        model.setListeners(new ScoreIterationListener(50));//new StatsListener(statsStorage));

        for ( int n = 0; n < nEpochs; ++n) //loops for the iterations
        {
            while(trainIter.hasNext())
            {
                model.fit(trainIter.next());
            }
            trainIter.reset();
        }

        Evaluation eval = new Evaluation(2);

        while(trainIter.hasNext()) //for the evaluation
        {
            DataSet dt = trainIter.next();

            eval.eval(model.output(dt.getFeatures()),model.getLabels()); //labels it
        }
        trainIter.reset(); //back to original data

        System.out.println(eval.stats());


        Path modelSaveAs = Paths.get(System.getProperty("java.io.tmpdir")  , "PredictGender.zip");
        ModelSerializer.writeModel(model, modelSaveAs.toString(),true); //after i run i save the result inside this

        System.out.println("Modelsave at " + modelSaveAs);

    }
}

