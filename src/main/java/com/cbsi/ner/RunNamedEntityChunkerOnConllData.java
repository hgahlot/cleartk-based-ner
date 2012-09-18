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
import java.util.Arrays;

import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.collection.CollectionReaderDescription;
import org.apache.uima.jcas.JCas;
import org.cleartk.classifier.CleartkSequenceAnnotator;
import org.cleartk.classifier.jar.DefaultSequenceDataWriterFactory;
import org.cleartk.classifier.jar.DirectoryDataWriterFactory;
import org.cleartk.classifier.jar.GenericJarClassifierFactory;
import org.cleartk.classifier.mallet.MalletCRFStringOutcomeDataWriter;
import org.cleartk.ne.type.NamedEntityMention;
import org.cleartk.syntax.opennlp.PosTaggerAnnotator;
import org.cleartk.syntax.opennlp.SentenceAnnotator;
import org.cleartk.token.tokenizer.TokenAnnotator;
import org.cleartk.util.Options_ImplBase;
import org.cleartk.util.ae.UriToDocumentTextAnnotator;
import org.cleartk.util.cr.UriCollectionReader;
import org.kohsuke.args4j.Option;
import org.uimafit.component.JCasAnnotator_ImplBase;
import org.uimafit.factory.AggregateBuilder;
import org.uimafit.factory.AnalysisEngineFactory;
import org.uimafit.pipeline.SimplePipeline;
import org.uimafit.util.JCasUtil;

import com.cbsi.ner.reader.Conll2003GoldReader;

/**
 * This class provides a main method that demonstrates how to run a trained
 * {@link NamedEntityChunker} on new files.
 * 
 * <br>
 * Copyright (c) 2012, Regents of the University of Colorado <br>
 * All rights reserved.
 * 
 * @author Steven Bethard
 */
public class RunNamedEntityChunkerOnConllData {

  public static class Options extends Options_ImplBase {
    @Option(name = "--model-dir", usage = "The directory where the model was trained")
    public File modelDirectory = new File("target/chunking/ne-model-comp-prod-noBI");

    @Option(name = "--test-file", usage = "The file to label with named entities.")
    public File testFile = new File("src/main/resources/data/cbsi-ner-data/test/cmp_prod_conll2003.test");
  }

  public static void main(String[] args) throws Exception {
    Options options = new Options();
    options.parseOptions(args);
    
    // a reader that loads the the CONLL 2003 format test file
    CollectionReaderDescription reader = Conll2003GoldReader.getDescription(options.testFile.getAbsolutePath());

    // assemble the testing pipeline
    AggregateBuilder aggregate = new AggregateBuilder();

    // an annotator that adds part-of-speech tags (so we can use them for features)
    aggregate.add(PosTaggerAnnotator.getDescription());

 // our NamedEntityChunker annotator, configured to classify on the new texts in conll 2003 format
    aggregate.add(AnalysisEngineFactory.createPrimitiveDescription(
        NamedEntityChunker.class,
        CleartkSequenceAnnotator.PARAM_IS_TRAINING,
        false,
        GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
        new File(options.modelDirectory, "model.jar")));

    // a very simple annotator that just prints out any named entities we found
    aggregate.add(AnalysisEngineFactory.createPrimitiveDescription(PrintNamedEntityMentions.class));

    // run the classification pipeline on the new texts
    SimplePipeline.runPipeline(reader, aggregate.createAggregateDescription());
  }

  /**
   * A simple annotator that just prints out any {@link NamedEntityMention}s in the CAS.
   * 
   * A real pipeline would probably decide on an appropriate output format and write files instead
   * of printing to standard output.
   */
  public static class PrintNamedEntityMentions extends JCasAnnotator_ImplBase {

    @Override
    public void process(JCas jCas) throws AnalysisEngineProcessException {
      for (NamedEntityMention mention : JCasUtil.select(jCas, NamedEntityMention.class)) {
        System.out.printf("%s (%s)\n", mention.getCoveredText(), mention.getMentionType());
      }
    }

  }
}
