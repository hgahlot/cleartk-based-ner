/** 
 * Copyright (c) 2007-2012, Regents of the University of Colorado 
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.FileFilterUtils;
import org.apache.commons.io.filefilter.HiddenFileFilter;
import org.apache.commons.io.filefilter.IOFileFilter;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.cas.CAS;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.resource.metadata.TypeSystemDescription;
import org.cleartk.classifier.CleartkSequenceAnnotator;
import org.cleartk.classifier.jar.DefaultSequenceDataWriterFactory;
import org.cleartk.classifier.jar.DirectoryDataWriterFactory;
import org.cleartk.classifier.jar.GenericJarClassifierFactory;
import org.cleartk.classifier.jar.Train;
import org.cleartk.classifier.mallet.MalletCRFStringOutcomeDataWriter;
import org.cleartk.eval.AnnotationStatistics;
import org.cleartk.eval.Evaluation_ImplBase;
import org.cleartk.ne.type.NamedEntityMention;
import org.cleartk.syntax.opennlp.PosTaggerAnnotator;
import org.cleartk.token.stem.snowball.DefaultSnowballStemmer;
import org.cleartk.token.type.Sentence;
import org.cleartk.type.test.Token;
import org.cleartk.util.Options_ImplBase;
import org.kohsuke.args4j.Option;
import org.uimafit.component.ViewTextCopierAnnotator;
import org.uimafit.factory.AggregateBuilder;
import org.uimafit.factory.AnalysisEngineFactory;
import org.uimafit.factory.TypeSystemDescriptionFactory;
import org.uimafit.pipeline.JCasIterable;
import org.uimafit.pipeline.SimplePipeline;
import org.uimafit.util.JCasUtil;

import com.cbsi.ner.RunNamedEntityChunker.PrintNamedEntityMentions;
import com.cbsi.ner.reader.Conll2003GoldReader;
import com.google.common.base.Function;

/**
 * <br>
 * Copyright (c) 2012, Regents of the University of Colorado <br>
 * All rights reserved.
 * <p>
 * This evaluator class provides a concrete example of how to train and evaluate classifiers.
 * Specifically this class will train a document categorizer using a subset of the 20 newsgroups
 * dataset. It evaluates performance using 2-fold cross validation as well as a holdout set.
 * <p>
 * 
 * Key points: <br>
 * <ul>
 * <li>Creating training and evaluator pipelines
 * <li>Example of feature transformation / normalization
 * </ul>
 * 
 * 
 * @author Lee Becker
 */
public class EvaluateNamedEntityChunker extends
Evaluation_ImplBase<File, AnnotationStatistics<String>> {

	public static class Options extends Options_ImplBase {
		@Option(
				name = "--train-dir",
				usage = "Specify the directory containing the training documents.  This is used for cross-validation, and for training in a holdout set evaluator. "
						+ "When we run this example we point to a directory containing training data from a subset of the 20 newsgroup corpus - i.e. a directory called '3news-bydate/train'")
		public File trainDirectory = new File("src/main/resources/data/cbsi-ner-data/train");

		@Option(
				name = "--test-dir",
				usage = "Specify the directory containing the test (aka holdout/validation) documents.  This is for holdout set evaluator. "
						+ "When we run this example we point to a directory containing training data from a subset of the 20 newsgroup corpus - i.e. a directory called '3news-bydate/test'")
		public File testDirectory = new File("src/main/resources/data/cbsi-ner-data/test");

		@Option(
				name = "--models-dir",
				usage = "specify the directory in which to write out the trained model files")
		public File modelsDirectory = new File("target/chunking/ne-model-feature-test");

		@Option(
				name = "--training-args",
				usage = "specify training arguments to be passed to the learner.  For multiple values specify -ta for each - e.g. '-ta -t -ta 0'")
		public List<String> trainingArguments = Arrays.asList("-t", "0");
	}

	public static enum AnnotatorMode {
		TRAIN, TEST, CLASSIFY
	}

	public static List<File> getFilesFromDirectory(File directory) {
		IOFileFilter fileFilter = FileFilterUtils.makeSVNAware(HiddenFileFilter.VISIBLE);
		IOFileFilter dirFilter = FileFilterUtils.makeSVNAware(FileFilterUtils.and(
				FileFilterUtils.directoryFileFilter(),
				HiddenFileFilter.VISIBLE));
		return new ArrayList<File>(FileUtils.listFiles(directory, fileFilter, dirFilter));
	}

	public static void main(String[] args) throws Exception {
		Options options = new Options();
		options.parseOptions(args);

		List<File> trainFiles = getFilesFromDirectory(options.trainDirectory);
		List<File> testFiles = getFilesFromDirectory(options.testDirectory);

		EvaluateNamedEntityChunker evaluator = new EvaluateNamedEntityChunker(
				options.modelsDirectory,
				options.trainingArguments);

		// Run Cross Validation
		//    List<AnnotationStatistics<String>> foldStats = evaluator.crossValidation(trainFiles, 2);
		//    AnnotationStatistics<String> crossValidationStats = AnnotationStatistics.addAll(foldStats);
		//
		//    System.err.println("Cross Validation Results:");
		//    System.err.print(crossValidationStats);
		//    System.err.println();
		//    System.err.println(crossValidationStats.confusions());
		//    System.err.println();

		// Run Holdout Set
		//AnnotationStatistics<String> holdoutStats = evaluator.trainAndTest(trainFiles, testFiles);
		AnnotationStatistics<String> holdoutStats = evaluator.test(
				Conll2003GoldReader.getCollectionReader(testFiles.get(0).getAbsolutePath()),
				new File(options.modelsDirectory+"/"));
		System.err.println("Holdout Set Results:");
		System.err.print(holdoutStats);
		System.err.println();
		System.err.println(holdoutStats.confusions());
	}

	public static final String GOLD_VIEW_NAME = "NamedEntityGoldView";

	public static final String SYSTEM_VIEW_NAME = CAS.NAME_DEFAULT_SOFA;

	private List<String> trainingArguments;

	public EvaluateNamedEntityChunker(File baseDirectory) {
		super(baseDirectory);
		this.trainingArguments = Arrays.<String> asList();
	}

	public EvaluateNamedEntityChunker(File baseDirectory, List<String> trainingArguments) {
		super(baseDirectory);
		this.trainingArguments = trainingArguments;
	}

	@Override
	protected CollectionReader getCollectionReader(List<File> items) throws Exception {
		// a reader that loads the the CONLL 2003 format train file
		return Conll2003GoldReader.getCollectionReader(items.get(0).getAbsolutePath());
	}

	@Override
	public void train(CollectionReader collectionReader, File outputDirectory) throws Exception {

		// Create and run the NER training pipeline
		AggregateBuilder builder = new AggregateBuilder();
		// an annotator that adds part-of-speech tags (so we can use them for features)
		builder.add(PosTaggerAnnotator.getDescription());
		builder.add(DefaultSnowballStemmer.getDescription("English"));

		// our NamedEntityChunker annotator, configured to write Mallet CRF training data
		builder.add(AnalysisEngineFactory.createPrimitiveDescription(
				NamedEntityChunker.class,
				CleartkSequenceAnnotator.PARAM_IS_TRAINING,
				true,
				DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
				outputDirectory,
				DefaultSequenceDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
				MalletCRFStringOutcomeDataWriter.class));

		// run the pipeline over the training corpus
		SimplePipeline.runPipeline(collectionReader, builder.createAggregateDescription());

		// train a Mallet CRF model on the training data
		Train.main(outputDirectory);

	}

	/**
	 * Creates the preprocessing pipeline needed for document classification. Specifically this
	 * consists of:
	 * <ul>
	 * <li>Populating the default view with the document text (as specified in the URIView)
	 * <li>Sentence segmentation
	 * <li>Tokenization
	 * <li>Stemming
	 * <li>[optional] labeling the document with gold-standard document categories
	 * </ul>
	 */
//	public static AggregateBuilder createPreprocessingAggregate(
//			File modelDirectory,
//			AnnotatorMode mode) throws ResourceInitializationException {
//		AggregateBuilder builder = new AggregateBuilder();
//
//		// Annotate documents with gold standard labels
//		switch (mode) {
//		case TRAIN:
//			// If this is training, put the label categories directly into the default view
//			//builder.add(AnalysisEngineFactory.createPrimitiveDescription(GoldDocumentCategoryAnnotator.class));
//			break;
//
//		case TEST:
//			// Copies the text from the default view to a separate gold view
//			builder.add(AnalysisEngineFactory.createPrimitiveDescription(
//					ViewTextCopierAnnotator.class,
//					ViewTextCopierAnnotator.PARAM_SOURCE_VIEW_NAME,
//					SYSTEM_VIEW_NAME,
//					ViewTextCopierAnnotator.PARAM_DESTINATION_VIEW_NAME,
//					GOLD_VIEW_NAME));
//
//			//        // If this is testing, put the document categories in the gold view
//			//        // The extra parameters to add() map the default view to the gold view.
//			//        builder.add(
//			//            AnalysisEngineFactory.createPrimitiveDescription(GoldDocumentCategoryAnnotator.class),
//			//            CAS.NAME_DEFAULT_SOFA,
//			//            GOLD_VIEW_NAME);
//			break;
//
//		case CLASSIFY:
//		default:
//			// In normal mode don't deal with gold labels
//			break;
//		}
//
//		return builder;
//	}

	/**
	 * Creates the aggregate builder for the document classification pipeline
	 */
//	public static AggregateBuilder createDocumentClassificationAggregate(
//			File modelDirectory,
//			AnnotatorMode mode) throws ResourceInitializationException {
//
//		AggregateBuilder builder = EvaluateNamedEntityChunker.createPreprocessingAggregate(
//				modelDirectory,
//				mode);
//
//		// an annotator that adds part-of-speech tags (so we can use them for features)
//		builder.add(PosTaggerAnnotator.getDescription());
//
//		switch (mode) {
//		case TRAIN:
//			break;
//		case TEST:
//		case CLASSIFY:
//		default:
//			// our NamedEntityChunker annotator, configured to classify on the new texts
//			builder.add(AnalysisEngineFactory.createPrimitiveDescription(
//					NamedEntityChunker.class,
//					CleartkSequenceAnnotator.PARAM_IS_TRAINING,
//					false,
//					GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
//					new File(modelDirectory, "model.jar")));
//			break;
//		}
//		return builder;
//	}

	@Override
	protected AnnotationStatistics<String> test(CollectionReader collectionReader, File directory)
			throws Exception {
		AnnotationStatistics<String> stats = new AnnotationStatistics<String>();
		
		TypeSystemDescription tsd = TypeSystemDescriptionFactory.createTypeSystemDescription();
		
		//create an AggregateBuilder (and then an AnalysisEngine) to identify
		//system NamedEntityMentions
		AggregateBuilder systemTaggingBuilder = new AggregateBuilder();
		//an annotator that adds part-of-speech tags (so we can use them for features)
		systemTaggingBuilder.add(PosTaggerAnnotator.getDescription());
		//an annotator that adds stem of token (so we can use it for features)
		systemTaggingBuilder.add(DefaultSnowballStemmer.getDescription("English"));
		systemTaggingBuilder.add(AnalysisEngineFactory.createPrimitiveDescription(
				NamedEntityChunker.class,
				CleartkSequenceAnnotator.PARAM_IS_TRAINING,
				false,
				GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
				new File(directory, "model.jar")));
		AnalysisEngine systemTaggingEngine = systemTaggingBuilder.createAggregate();

		Function<NamedEntityMention, ?> getSpan = AnnotationStatistics.annotationToSpan();
		Function<NamedEntityMention, String> getCategory = AnnotationStatistics.annotationToFeatureValue("mentionType");
		
		//loop to run the collection reader to read in gold NamedEntityMention 
		//annotations (apart from other annotations) and then run the systemTaggingEngine
		//to identify NamedEntityMentions using our NamedEntityChunker
		for (JCas jCas : new JCasIterable(collectionReader, tsd)) {
			//store gold NamedEntityMention annotations
			List<NamedEntityMention> goldNem = new ArrayList<NamedEntityMention>(JCasUtil.select(jCas, NamedEntityMention.class));
			
			//remove the gold annotations from the jCas so that 
			//the NamedEntityChunker can be run to populate system
			//NamedEntityAnnotations in the same jCas
			for(NamedEntityMention nm: goldNem){
				try {
					nm.removeFromIndexes(jCas);
				} catch (Exception ex){
					System.out.println("Exception in named entity removal from jCas");
					continue;
				}
			}
			
			//now run the AnalysisEngine for named entity chunking to 
			//populate NamedEntityMentions in the same jCas
			SimplePipeline.runPipeline(jCas, systemTaggingEngine);
			
			//store system identfied NamedEntityMentions
			List<NamedEntityMention> systemNem = new ArrayList<NamedEntityMention>(JCasUtil.select(jCas, NamedEntityMention.class));
			
			//add the stats obtained from current jCas to a global 
			//AnnotationStatistics object
			stats.add(goldNem, systemNem, getSpan, getCategory);
		}

		return stats;
	}
}