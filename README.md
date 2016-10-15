# Introduction to the Deeplearning4J API 

## I. GETTING TO KNOW ND4J (numpy for java)

The main objective here is to familiarize oneself with the nd4j api.  
NDArrays are n-dimensional arrays. Datasets, weights/biases are all ultimately build with NDArrays.  
Obviously these questions are somewhat contrived but the goal here is to get thoroughly comfortable with the API.  

References:  
    Excellent exercises by Alex in the dl4j-exercises repo  
    https://github.com/deeplearning4j/dl4j-examples/tree/master/nd4j-examples/src/main/java/org/nd4j/examples  
    Also these -  
    http://nd4j.org/documentation  
    http://nd4j.org/userguide  

```java
public class Nd4jExercises {

    public static void main(String[] args) {
        /*==============================================================================================
          Exercise 0: Accessing 2d, 3d and 4d matrices
            0a. Generate a random 2d NDarray of size mxn,say "A" where the rows are multiples of each other(1,2,..n)
            * Shuffle this matrix row wise and save it to "A"
            * Reduce this matrix to a column vector,say "C" where each element is the average of a row.
            * Find the min of the column vector "C" and divide every element by it.
            * Sort this vector and  it is equal to Nd4j.linspace(1,n).transpose()

            0b. Generate a random 3d NDarray of size lxmxn. 3d NDArrays come up when doing RNNS.
            The input to an RNN is 3d - samples x features x timesteps.
            Take NOTE of this Order.
            * Print out the first sample here (so the very first mxn matrices).
            * Print out all the timesteps for every samples for feature 0.
            * What will max along the 2nd dimension give you here? Size of this array?
            * What will max along the 0 and 1st dimension give you here? Size of this array?
            Bonus: Generate a 3d array where the timesteps are consecutive numbers and the features are multiples of each other.

            0c. Generate a random 4d NDArray of kxlxmxn. 4d NDArrays come up when doing CNNs.
            The input to a CNN is 4d - sample x channel x height x width
            * Print out the zeroth sample (so k lxmxn matrices)
            * Print out the second channel of every image (index 1)
            * Transform this kxlxmxn array to kx1xmxn array. So just a single channel which is the average of all "l" channels

            0d. Can you make an NDarray of size = {1}? How about size = {5}?
        ===============================================================================================*/
        //Answers:

        exerciseZeroa();
        exerciseZerob();
        exerciseZeroc();
        exerciseZerod();

        /*==============================================================================================
        Exercise 1: Understanding views
            1a. Generate a random nxm NDArray say "A". Select it's odd numbered columns and write them to a new array "AA".
            Add 10 to all the elements to AA. Now print A and AA. What happened to A?

            1b. What happens when you try to do A.getRow(0).asDouble()? How many values is that returning?

            1c. What if I only want to get double values for the 0th row.
       ===============================================================================================*/
        //Answers:

        exerciseOnea();
        exerciseOneb();
        exerciseOnec();

       /*==============================================================================================
       BONUS Exercises:
       Transforms, Accumulations, Broadcasting and Misc...
            2a. Set data type to double.
            2b. Generate a random NDArray mxn A
            2c. Write it to a file.
            2d. Read it back in as another array B
            2e. Assert A == B
            2e. Make another array allTwoPi = Nd4j.valueArrayOf(1, n, 2*pi))
            2f. Add (inplace) allTwoPi to every row in A
            2e. Take the cosine similarity between Cos(A) and Cos(B) and write to A. Assert A is all 1s.
            2f. Accumulate A to be sum of it's elements. Assert it's equal to m. Watch for double vs float vs int!

        For the curious: Understanding Tensor Along Dimensions
        {not necessary - unless you dream of ndarrays and want to contribute to nd4j}
        http://nd4j.org/userguide#getsettensor
       ================================================================================================*/
    }

}
```

## II. BUILD YOUR OWN MLP - FIZZBUZZ
This exercise follows this example pretty closely. Use it for reference! 
https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4jexamples/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierLinear.java

```java
public class FizzBuzzMLP {


    public static void main(String[] args) throws Exception {
        /*=========================================================
          STEP0: Run through the MLP linear classifier
                        from the example repo if you haven't already
        ==========================================================*/

        //0a. declare necessary variables


        /*=========================================================
            STEP1: Load csv into a dataset/datasetiterator
            Resources(in addition to example repo):
                https://deeplearning4j.org/csv-deep-learning
        ==========================================================*/

        //1a. Set up training dataset iterator

        //1b. Set up test/evaluation dataset iterator

        //1c. Print some entries to see??

        /*=========================================================
                       STEP2: Setup your NN
        ==========================================================*/

        //2a. Set up configuration

        //2b. From configuration to model

        //2c. Initialize model and set up listeners - score and histogram


        /*=========================================================
                       STEP3: Train your NN
        ==========================================================*/

        //3a. Fit your model with the training data 

        /*=========================================================
                       STEP4: Evaluate your NN
        ==========================================================*/

        //4a. Evaluate your model with the test data

        //4b. Print evaluation statistics

        /*=========================================================
                       STEP4: Tune your NN
            Resources(in addition to example repo):
                Today's slides on tuning
                Live support: gitter tuning channel!
         https://gitter.im/deeplearning4j/deeplearning4j/tuninghelp
        ==========================================================*/


    }


}
```

## III. GO THE EXTRAMILE - FIZZBUZZ as a sequence (1,15), Build an RNN to memorize this sequence

You can use this example for reference which is very similar (memorize a string)  
https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/basic/BasicRNNExample.java  

```java
public class FizzBuzzRNN {

}
```
