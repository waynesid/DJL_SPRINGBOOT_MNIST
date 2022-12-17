package com.AthenaML.DJL001.Service;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.nn.Activation;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import com.AthenaML.DJL001.MultiLayerPerceptron;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Wayne Sidney
 * Created on {18/12/2022}
 */
@Service
public class TestingServiceImpl implements TestingService{
    @Override
    public Classifications.Classification testDataset() {
        Image image = ImageFactory.getInstance().fromFile(Paths.get(IMAGE_FILE));
        try(Model model = Model.newInstance(MODEL_NAME)){
            model.setBlock(new MultiLayerPerceptron(Mnist.IMAGE_HEIGHT*Mnist.IMAGE_WIDTH, Mnist.NUM_CLASSES, new int[]{128,64}, Activation::sigmoid));
            model.load(Paths.get(MODEL_DIRE));
            List<String> classes = IntStream.range(0,10)
                    .mapToObj(String::valueOf)
                    .collect(Collectors.toList());
            ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                    .addTransform(new ToTensor())
                    .optSynset(classes)
                    .build();

            try(Predictor<Image, Classifications> predictor =
                    model.newPredictor(translator)){
                return predictor.predict(image).best();
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }

        }
        return null;
    }
}
