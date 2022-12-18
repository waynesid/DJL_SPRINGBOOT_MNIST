package com.AthenaML.DJL001.Service;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.translate.TranslateException;
import com.AthenaML.DJL001.Helper.HelperService;
import com.AthenaML.DJL001.MultiLayerPerceptron;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * @author Wayne Sidney
 * Created on {18/12/2022}
 */
@Service
public class TrainingServiceImpl implements TrainingService{

    private static final String MODEL_NAME = "model_mnistClassification";

    private static final String MODEL_DIRE = "~/AthenaMLJava/DJL-001/src/main/java/com/AthenaML/DJL001/Djl001Application.java";
    private static final int BATCH_SIZE = 32;
    private static final int LIMIT = 40;
    private static final int EPOCH = 25;
    @Autowired
    private HelperService helperService;
    @Override
    public TrainingResult trainMnistDataSet() throws IOException {
        try(Model model = Model.newInstance(MODEL_NAME)){

            model.setBlock(new MultiLayerPerceptron
                    (Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH, Mnist.NUM_CLASSES, new int[]{128, 64}, Activation::sigmoid));
            RandomAccessDataset trainingSet = helperService.getDataSet(Dataset.Usage.TRAIN, BATCH_SIZE, LIMIT);
            RandomAccessDataset validateSet = this.helperService.getDataSet(Dataset.Usage.TEST, BATCH_SIZE, LIMIT);

            try (Trainer trainer = model.newTrainer(this.helperService.constructTrainingConfig(MODEL_NAME))) {
                trainer.setMetrics(new Metrics());
                trainer.initialize(new Shape(1, Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH));
                EasyTrain.fit(trainer, EPOCH, trainingSet, validateSet);
                Path modelDire = Paths.get(MODEL_DIRE);
                Files.createDirectories(modelDire);
                model.setProperty("Epoch", String.valueOf(EPOCH));
                model.save(modelDire, MODEL_NAME);
                return trainer.getTrainingResult();
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
