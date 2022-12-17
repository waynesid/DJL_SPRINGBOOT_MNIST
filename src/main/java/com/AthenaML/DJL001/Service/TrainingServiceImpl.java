package com.AthenaML.DJL001.Service;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.nn.Activation;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import com.AthenaML.DJL001.Helper.HelperService;
import com.AthenaML.DJL001.MultiLayerPerceptron;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * @author Wayne Sidney
 * Created on {18/12/2022}
 */
@Service
public class TrainingServiceImpl implements TrainingService{
    @Autowired
    private HelperService helperService;
    @Override
    public TrainingResult trainMnistDataSet() {
        //add a MODEL_NAME in the newInstance() method as parameter
        try(Model model = Model.newInstance() {
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
            }
        }
    }
}
