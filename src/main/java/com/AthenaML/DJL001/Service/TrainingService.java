package com.AthenaML.DJL001.Service;

import ai.djl.training.TrainingResult;

import java.io.IOException;

/**
 * @author Wayne Sidney
 * Created on {18/12/2022}
 */
public interface TrainingService {
    public TrainingResult trainMnistDataSet() throws IOException;
}
