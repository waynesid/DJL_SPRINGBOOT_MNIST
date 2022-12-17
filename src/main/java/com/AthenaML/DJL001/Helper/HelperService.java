package com.AthenaML.DJL001.Helper;

import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;

import java.io.IOException;

/**
 * @author Wayne Sidney
 * Created on {18/12/2022}
 */
public interface HelperService {
    public RandomAccessDataset getDataSet(Dataset.Usage usage, int batchSize, int limit) throws IOException;

    public DefaultTrainingConfig constructTrainingConfig(String outputDire);
}
