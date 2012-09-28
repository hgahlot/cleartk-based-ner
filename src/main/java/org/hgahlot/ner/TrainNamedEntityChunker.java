/* 
 * Copyright (c) 2012, Regents of the University of Colorado 
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
 * Neither the name of the University of Colorado at Boulder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE. 
 */
package com.cbsi.ner;

import java.io.File;

import org.apache.commons.io.filefilter.IOFileFilter;
import org.apache.uima.collection.CollectionReaderDescription;
import org.cleartk.classifier.CleartkSequenceAnnotator;
import org.cleartk.classifier.jar.DefaultSequenceDataWriterFactory;
import org.cleartk.classifier.jar.DirectoryDataWriterFactory;
import org.cleartk.classifier.jar.Train;
import org.cleartk.classifier.mallet.MalletCRFStringOutcomeDataWriter;
import org.cleartk.syntax.opennlp.PosTaggerAnnotator;
import org.cleartk.token.stem.snowball.DefaultSnowballStemmer;
import org.cleartk.util.Options_ImplBase;
import org.kohsuke.args4j.Option;
import org.uimafit.factory.AggregateBuilder;
import org.uimafit.factory.AnalysisEngineFactory;
import org.uimafit.pipeline.SimplePipeline;

import com.cbsi.ner.reader.Conll2003GoldReader;

/**
 * This class provides a main method that demonstrates how to train a {@link NamedEntityChunker} on
 * an annotated corpus of named entities.
 * 
 * <br>
 * Copyright (c) 2012, Regents of the University of Colorado <br>
 * All rights reserved.
 * 
 * @author Steven Bethard
 */
public class TrainNamedEntityChunker {

  public static class Options extends Options_ImplBase {
    @Option(name = "--train-dir", usage = "The directory containing Conll-annotated files")
    public File trainFile = new File(
        "src/main/resources/data/cbsi-ner-data/cmp_prod_conll2003.train");

    @Option(name = "--model-dir", usage = "The directory where the model should be written")
    public File modelDirectory = new File("target/chunking/ne-model-feature-test");
  }

  public static void main(String[] args) throws Exception {
    Options options = new Options();
    options.parseOptions(args);

    // a reader that loads the the CONLL 2003 format train file
    CollectionReaderDescription reader = Conll2003GoldReader.getDescription(options.trainFile.getAbsolutePath());

    // assemble the training pipeline
    AggregateBuilder aggregate = new AggregateBuilder();

    // an annotator that adds part-of-speech tags (so we can use them for features)
    aggregate.add(PosTaggerAnnotator.getDescription());
    
    // an annotator that adds the stem of the token (so we can use them for features)
    aggregate.add(DefaultSnowballStemmer.getDescription("English"));

    // our NamedEntityChunker annotator, configured to write Mallet CRF training data
    aggregate.add(AnalysisEngineFactory.createPrimitiveDescription(
        NamedEntityChunker.class,
        CleartkSequenceAnnotator.PARAM_IS_TRAINING,
        true,
        DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
        options.modelDirectory,
        DefaultSequenceDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
        MalletCRFStringOutcomeDataWriter.class));

    // run the pipeline over the training corpus
    SimplePipeline.runPipeline(reader, aggregate.createAggregateDescription());

    // train a Mallet CRF model on the training data
    Train.main(options.modelDirectory);
  }

  /**
   * An auxiliary class necessary to only load the ".txt" files from the MASC directories.
   * 
   * You can mostly ignore this - it's only necessary due to the idiosyncracies of the MASC
   * directory structure.
   */
  public static class MASCTextFileFilter implements IOFileFilter {
    public boolean accept(File file) {
      return file.getPath().endsWith(".txt");
    }

    public boolean accept(File dir, String name) {
      return name.endsWith(".txt");
    }
  }

}
