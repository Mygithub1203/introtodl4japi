package org.nd4j.exercises;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;


/**
 * Created by susaneraly on 10/14/16.
 */

/*
    The main objective here is to familiarize oneself with the nd4j api.
    NDArrays are n-dimensional arrays. Datasets, weights/biases are all ultimately
    build with NDArrays.
    Obviously these questions are construed, but I find yourself relying on
    operations like this when I am writing tests for new functionality I have added.

    References:
    Excellent exercises by Alex in the dl4j-exercises repo
	https://github.com/deeplearning4j/dl4j-examples/tree/master/nd4j-examples/src/main/java/org/nd4j/examples
    Also these -
    http://nd4j.org/documentation
    http://nd4j.org/userguide

 */
public class Nd4jExercises {
    public static int k = 6;
    public static int l = 2;
    public static int m = 4;
    public static int n = 3;

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
            2e. Make another array allTwoPi = Nd4j.valueArrayOf(1, n, Math.constants(2*pi))
            2f. Add (inplace) allTwoPi to every row in A
            2e. Take the cosine similarity between A and B and write to A. Assert A is all 1s.
            2f. Accumulate A to be sum of it's elements. Assert it's equal to m. Watch for double vs float vs int!

        For the curious: Understanding Tensor Along Dimensions
        {not necessary - unless you dream of ndarrays and want to contribute to nd4j}
        http://nd4j.org/userguide#getsettensor
       ================================================================================================*/
    }

    public static void exerciseZeroa() {
        //Answers:
        INDArray A = Nd4j.ones(m,n);
        INDArray tmp = Nd4j.rand(1,n);
        A.muliRowVector(tmp);
        tmp = Nd4j.linspace(1,m,m);
        System.out.println(ArrayUtils.toString(tmp.shape())); //linspace gives a row vector
        tmp = tmp.transpose(); //transposei and transpose neither are in place - current master, workaround do this
        System.out.println(ArrayUtils.toString(tmp.shape())); //linspace gives a row vector
        A.muliColumnVector(tmp);
        System.out.println(A);
        Nd4j.shuffle(A,1); //shuffle operation is the dimension to keep intact here so dimension 1. I get confused too
        System.out.println(A);

        A = A.mean(1); //taking mean along the columns, therefore dimension 1
        System.out.println(ArrayUtils.toString(A.shape())); //checking shape after mean
        float minHere = A.min(0).getFloat(0); //.min(0) returns an NDArray convert that to float
        A.divi(minHere);
        System.out.println(A); //Before sort
        A = Nd4j.sort(A,0,true);
        System.out.println(A); //After sort
        System.out.println(ArrayUtils.toString(A.shape())); //Don't be fooled by the print A is a column vector

        tmp = Nd4j.linspace(1,m,m);
        System.out.println(A.equals(tmp));
        tmp = tmp.transpose();
        System.out.println(A == tmp);//This is false, why?
        System.out.println(A.eq(tmp)); //This gives all 1s, because it does element wise comparisons
        // I want a boolean, now what??
        System.out.println(A.equals(tmp));
    }

    public static void exerciseZerob() {
        //Printing with 3d input - samples x features x timesteps.
        INDArray A = Nd4j.rand(l*m,n).reshape(l,m,n);
        System.out.println("Printing A for context...");
        System.out.println(ArrayUtils.toString(A.shape())); //checking shape after mean
        System.out.println(A);
        INDArray tmp;
        System.out.println("Via slice..");
        tmp = A.slice(0,0);
        System.out.println(ArrayUtils.toString(tmp.shape()));
        System.out.println(tmp);
        System.out.println("Via NDArrayIndex..");
        tmp = A.get(NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.all());
        System.out.println(ArrayUtils.toString(tmp.shape()));
        System.out.println(tmp);
        System.out.println("Via tensor along dimension..");
        tmp = A.tensorAlongDimension(0,1,2);
        System.out.println(ArrayUtils.toString(tmp.shape()));
        System.out.println(tmp);

        //Print out all the timesteps for every samples for feature 0.
        //Again - samples x features x timesteps

        tmp = A.get(NDArrayIndex.all(),NDArrayIndex.point(0),NDArrayIndex.all());
        System.out.println("Printing A for context...");
        System.out.println(ArrayUtils.toString(A.shape())); //checking shape after mean
        System.out.println(A);
        System.out.println("This is all the timesteps for every sample for a single feature");
        System.out.println(tmp);

        //Max along second dimension
        tmp = A.max(2); //taking max along timesteps
        System.out.println("This is the max along dimension 2 or timesteps: shape and values");
        System.out.println(ArrayUtils.toString(tmp.shape()));
        System.out.println(tmp);


        //Max along zero and 1st dimension
        tmp = A.max(1,0); //taking max along timesteps
        System.out.println("This is the max along dimension 1 & 0 or samples & features: shape and values");
        System.out.println(ArrayUtils.toString(tmp.shape()));
        System.out.println(tmp);
        //what if I do max(0,1) - no difference
        tmp = A.max(0,1); //taking max along timesteps
        System.out.println("This is the max along dimension 0 & 1 or samples & features: shape and values");
        System.out.println(ArrayUtils.toString(tmp.shape()));
        System.out.println(tmp);
    }

    public static void exerciseZeroc() {
        //we are talking images now: 4d sample x channel x height x columns
        INDArray A = Nd4j.rand(k*l*m,n).reshape(k,l,m,n);

        //print out zeroth sample - same as above, you can use many different methods. I like tensors.
        System.out.println("Printing A for context");
        System.out.println(A);
        System.out.println("Printing zeroth image");
        System.out.println(A.tensorAlongDimension(0,1,2,3));

        //print out the zeroth channel of every image
        //again many methods but I like to use tensors
        System.out.println("Printing A for context");
        System.out.println(A);
        System.out.println("The second channel (index-1) along all images - shape and values");
        System.out.println(ArrayUtils.toString(A.tensorAlongDimension(1,0,2,3).shape()));
        System.out.println(A.tensorAlongDimension(1,0,2,3));

        //Transform to single channel - mean of all channels
        System.out.println("Converting to single channel: shape and vals");
        System.out.println(ArrayUtils.toString(A.mean(1).shape()));
        System.out.println(A.mean(1));
        System.out.println("NOTE: shape is not rank 4....Reshape if you want rank 4");
        System.out.println(A.mean(1).reshape(k,1,m,n));
    }

    public static void exerciseZerod() {
        System.out.println("NDArrays are by definition rank 2 or greater");
        System.out.println("Defaults to a row vector");
    }


    public static void exerciseOnea() {
        //1a. Generate a random nxm NDArray say "A". Add 10 to all the elements that are ONLY in odd numbered columns.
        INDArray A = Nd4j.rand(n,m);
        System.out.println(A);
        INDArray AA = A.get(NDArrayIndex.all(),NDArrayIndex.interval(1,2,m,true));
        AA.addi(10);
        System.out.println(AA);
        System.out.println("What happened to A...");
        System.out.println(A);
        System.out.println("Is AA a view??");
        System.out.println(AA.isView());
    }

    public static void exerciseOneb() {
        INDArray A = Nd4j.rand(n,m);
        System.out.println(ArrayUtils.toString(A.shape()));
        System.out.println(A);
        System.out.println(ArrayUtils.toString(A.getRow(0).shape()));
        System.out.println(ArrayUtils.toString(A.getRow(0).data().asDouble()));
    }

    public static void exerciseOnec() {
        INDArray A = Nd4j.rand(n,m);
        System.out.println(A);
        INDArray AA = A.get(NDArrayIndex.all(),NDArrayIndex.interval(1,2,m,true));
        System.out.println(ArrayUtils.toString(AA.getRow(0).data().asDouble()));
        System.out.println("Note how the below still does a dup of the original array...");
        System.out.println(ArrayUtils.toString(A.getRow(0).dup().data().asDouble()));
    }
}
