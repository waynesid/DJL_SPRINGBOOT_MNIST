package com.AthenaML.DJL001.Service;

import ai.djl.MalformedModelException;
import ai.djl.modality.Classifications;

import java.io.IOException;

/**
 * @author Wayne Sidney
 * Created on {18/12/2022}
 */
public interface TestingService {
    public Classifications.Classification testDataset() throws MalformedModelException, IOException;
}
