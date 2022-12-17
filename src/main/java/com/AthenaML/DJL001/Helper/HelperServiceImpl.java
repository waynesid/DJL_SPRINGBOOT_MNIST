package com.AthenaML.DJL001.Helper;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.engine.Engine;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import org.springframework.stereotype.Service;

import java.io.IOException;

/**
 * @author Wayne Sidney
 * Created on {18/12/2022}
 */
@Service
public class HelperServiceImpl implements HelperService{
    @Override
    public RandomAccessDataset getDataSet(Dataset.Usage usage, int batchSize, int limit) throws IOException {
        Mnist mnist = Mnist.builder()
                .optUsage(usage)
                .optLimit(limit)
                .setSampling(batchSize, true)
                .build();
        mnist.prepare(new ProgressBar());
        return mnist;
    }

    @Override
    public DefaultTrainingConfig constructTrainingConfig(String outputDire) {
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDire);
        listener.setSaveModelCallback(trainer -> {
            TrainingResult result = trainer.getTrainingResult();
            Model model = trainer.getModel();
            float accuracy = result.getTrainEvaluation("Accuracy");
            model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
            model.setProperty("Accuracy", String.format("%.5f", accuracy));
            new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .addEvaluator(new Accuracy())
                    .optDevices(Engine.getInstance().getDevices(Engine.getInstance().getGpuCount()))
                    .addTrainingListeners(TrainingListener.Defaults.logging(outputDire))
                    .addTrainingListeners(listener);
        });
        return null;
    }
}
