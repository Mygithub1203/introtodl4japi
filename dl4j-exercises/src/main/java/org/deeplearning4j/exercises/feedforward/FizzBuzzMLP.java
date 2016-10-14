package org.deeplearning4j.exercises.feedforward;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by susaneraly on 10/12/16.
 */
public class FizzBuzzMLP {
    private static Logger log = LoggerFactory.getLogger(FizzBuzzMLP.class);

    public static void main(String[] args) throws Exception {
        /*=========================================================
          STEP0: Run through the MLP linear classifier
                        from the example repo
            This exercise follows that example pretty closely.
		https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierLinear.java
        ==========================================================*/

        //0a. declare final variables and class?? variables
        int numInputs = 10;
        int numOutputs = 4;

        int numHiddenNodes = 200;
        //int numHiddenNodes = 100;

        double learningRate = 0.01;
        int nEpochs = 1000;
        //int nEpochs = 10000;
        int iterations = 1;
        int seed = 123;
        //int batchSize = 128;
        int batchSize = 64;
        //int batchSize = 256;

        /*=========================================================
            STEP1: Load csv into a dataset/datasetiterator
            Resources(in addition to example repo):
                https://deeplearning4j.org/csv-deep-learning
        ==========================================================*/

        //1a. Set up training dataset iterator
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("fizzbuzztrain.csv")));
        //RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,10,4);

        //1b. Set up test/evaluation dataset iterator
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("fizzbuzztest.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,10,4);

        //1c. Print some entries to see??
        DataSet next = trainIter.next(10);
        System.out.println(next.get(0));
        System.out.println(next.getFeatures());
        System.out.println(next.getLabels());

        System.out.println(next.get(2).getFeatures());
        System.out.println(next.get(2).getLabels());

        trainIter.reset();

        /*=========================================================
                       STEP2: Setup your NN
        ==========================================================*/

        //2a. Set up configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .updater(Updater.RMSPROP)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
            /*.layer(1, new DenseLayer.Builder().nIn(numHiddenNodes/2).nOut(numHiddenNodes/4)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build()) */
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .weightInit(WeightInit.XAVIER)
                .activation("softmax").weightInit(WeightInit.XAVIER)
                .nIn(numHiddenNodes).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();


        //2b. From configuration to model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);


        //2c. Initialize model and set up listeners - score and histogram
        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates
        model.setListeners(new HistogramIterationListener(1));


        /*=========================================================
                       STEP3: Train your NN
        ==========================================================*/

        //3a. Fit your model with the training data
        for ( int n = 0; n < nEpochs; n++) {
            while (trainIter.hasNext()) {
                DataSet tr = trainIter.next();
                model.fit(tr);
            }
            trainIter.reset();
        }

        /*=========================================================
                       STEP4: Evaluate your NN
        ==========================================================*/

        //4a. Evaluate your model with the test data
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(lables, predicted);

        }
        //A less verbose method here is:
        //model.evaluate(testIter);

        //4b. Print evaluation statistics
        System.out.println(eval.stats());

        /*=========================================================
                       STEP4: Tune your NN
            Resources(in addition to example repo):
                Today's slides on tuning
                Live support: gitter tuning channel!
         https://gitter.im/deeplearning4j/deeplearning4j/tuninghelp
        ==========================================================*/


    }


}
