package com.AthenaML.DJL001;

import ai.djl.ndarray.NDList;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;

import java.util.function.Function;

/**
 * @author Wayne Sidney
 * Created on {18/12/2022}
 */
public class MultiLayerPerceptron extends SequentialBlock {
    public void MultilayerPerceptron(int input, int output, int[] hidden, Function<NDList,NDList> activation){
        add(Blocks.batchFlattenBlock(input));
        for(int hiddenSize : hidden){
            add(Linear.builder().setUnits(hiddenSize).build());
            add(activation);
        }
        add(Linear.builder().setUnits(output).build());
    }
}
