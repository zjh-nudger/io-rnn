import java.util.*;
import java.io.*;
import org.apache.commons.exec.*;

public class UnsupParser {

	Reranker reranker;
	String mstPath = "/datastore/phong/data/io-rnn/tools/mstparser/";
	int K = 10;
	double alpha = 0;

	public void execute(String cmd, String outputFile) {
		try {
			System.out.println(cmd);
		    ByteArrayOutputStream stdout = new ByteArrayOutputStream();
    		PumpStreamHandler psh = new PumpStreamHandler(stdout);
	    	CommandLine cl = CommandLine.parse(cmd);
	    	DefaultExecutor exec = new DefaultExecutor();
		    exec.setStreamHandler(psh);
    		exec.execute(cl);
		    BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
			writer.write(stdout.toString());
			writer.close();
		} 
		catch (Exception e) {
		}
	} 

	public void execute(String cmd) {
		try {
			System.out.println(cmd);
	    	CommandLine cl = CommandLine.parse(cmd);
	    	DefaultExecutor exec = new DefaultExecutor();
	   		exec.execute(cl);
		} 
		catch (Exception e) {
		}

	}

	public UnsupParser(String dicPath, double alpha, int K) throws IOException {
		this.alpha = alpha;
		this.K = K;

		Lookup vocaDic = new Lookup(new Lookup.CollobertNormalizer()); 
		vocaDic.loadFromFile(dicPath + "/words.lst");
		if (!vocaDic.str2id.containsKey("UNKNOWN"))
			throw new Error ("not contain UNKNOWN");
		if (!vocaDic.str2id.containsKey("EOS")) 
			throw new Error("not contain EOS");

		Lookup posDic = new Lookup(new Lookup.NoNormalizer()); 
		posDic.loadFromFile(dicPath + "/pos.lst");
		if (!posDic.str2id.containsKey("EOS")) 
			throw new Error("not contain EOS");
		if (!posDic.str2id.containsKey("EOC")) 
			throw new Error("not contain EOC");
	
		Lookup deprelDic = new Lookup(new Lookup.NoNormalizer()); 
		deprelDic.loadFromFile(dicPath + "/deprel.lst");
		if (!deprelDic.str2id.containsKey("EOC")) 
			throw new Error("not contain EOC");
		
		this.reranker = new Reranker(vocaDic, posDic, deprelDic);
	}

	public void train(String modelDir, String traindsbankPath, String traingolddsPath, int nEpochs) throws Exception {
		String execMST = "java -classpath \"" + mstPath + ":" + mstPath + "lib/trove.jar\" " +
		                "-Xmx32g -Djava.io.tmpdir=./ mstparser.DependencyParser ";

    	execute("mkdir " + modelDir);
		execute("mkdir " + modelDir + "/warm_up/");
		
		String tempFile = modelDir + "/temp";

		// train MSTparser with dsbank
		String mstDir = modelDir + "/MST-1/";
		execute("mkdir " + mstDir);
		execute("cp " + traindsbankPath + " " + mstDir + "train.conll");
    	
		traindsbankPath = mstDir + "train.conll";
	    String trainkbestdsbankPath = mstDir + "train-" + String.valueOf(K) + "-best-mst2ndorder.conll";

	    execute(execMST + 
				" train train-file:" + traindsbankPath + " training-k:5 order:2 loss-type:nopunc" + 
				" model-name:" + mstDir + "/model " + 
				" test test-file:" + traindsbankPath + " testing-k:" + String.valueOf(K) + " output-file:" + trainkbestdsbankPath);
		execute("cp " + trainkbestdsbankPath + " " + tempFile);
	    execute("sed 's/<no-type>/NOLABEL/g' " + tempFile, trainkbestdsbankPath);


	    for (int it = 1; it <= nEpochs; it++) {
			System.out.println("================================== iter " + String.valueOf(it) + " ============================");
    	    String submodelDir = modelDir + "/" + it + "/";
        	execute("mkdir " + submodelDir);

	        // train reranker
			String warmUpDir = submodelDir + "/warm_up/";
            execute("mkdir " + warmUpDir);
			reranker.train(traindsbankPath);

			// training MSTParser
	        System.out.println("load train dsbank " + traindsbankPath + " " + trainkbestdsbankPath);
	        mstDir = modelDir + "/MST-" + (it+1) + "/";
			execute("mkdir " + mstDir);
			traindsbankPath = mstDir + "train.conll";
   			reranker.evaluate(trainkbestdsbankPath, traingolddsPath, traindsbankPath, alpha, K);

			trainkbestdsbankPath = mstDir + "train-" + String.valueOf(K) + "-best-mst2ndorder.conll";

		    execute(execMST + 
					" train train-file:" + traindsbankPath + " training-k:5 order:2 loss-type:nopunc" + 
					" model-name:" + mstDir + "/model " + 
					" test test-file:" + traindsbankPath + " testing-k:" + String.valueOf(K) + " output-file:" + trainkbestdsbankPath);
			execute("cp " + trainkbestdsbankPath + " " + tempFile);
		    execute("sed 's/<no-type>/NOLABEL/g' " + tempFile, trainkbestdsbankPath);
		}
	}

	public static void main(String[] args) throws Exception {
		
		String dicPath = "../data/wsj10/dic/"; 
		String dataPath = "../data/wsj10/data/"; 
		String modelDir = "model";
		String trainFile = "train15.udpc.conll";
		String goldFile = "train15.gold.conll";
		double alpha = 0;
		int K = 10;

		if (args.length == 7) {
			dicPath = args[0];
			dataPath = args[1];
			trainFile = args[2];
			goldFile = args[4];
			modelDir = args[5];
			alpha = Double.valueOf(args[5]);
			K = Integer.valueOf(args[6]);
		}
		else if (args.length > 0) {
			System.err.println("dicPath dataPath trainFile kbestFile gold_file alpha K");
			return;
		}

		UnsupParser parser = new UnsupParser(dicPath, alpha, K);
		parser.train(modelDir, dataPath + trainFile, dataPath + goldFile, 1000);
	}

}
