import java.io.IOException;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class Eval {

	public static Map<String, Integer> computeScores(DependencyStructure test, DependencyStructure gold) {
		if (test.items.length != gold.items.length)
			throw new Error("length not match");
		
		Map<String, Integer> ret = new HashMap<String, Integer>();
		int label = 0;
		int unlabel = 0;
		
		for (int i = 1; i < test.items.length; i++) {
			if (test.items[i].headPosition == gold.items[i].headPosition) {
				unlabel++;
				if (test.items[i].deprel == gold.items[i].deprel)
					label++;
			}
		}
		
		ret.put("label", label);
		ret.put("unlabel", unlabel);
		return ret;
	}
	
	public static Map<String, Double> computeScores(List<DependencyStructure> testbank, 
			List<DependencyStructure> goldbank) {
		if (testbank.size() != goldbank.size())
			throw new Error("size not match");
		
		Map<String, Double> ret = new HashMap<String, Double>();
		int label = 0;
		int unlabel = 0;
		int umatch = 0;
		int lmatch = 0;
		int total = 0;
		
		for (int i = 0; i < testbank.size(); i++) {
			DependencyStructure test = testbank.get(i);
			DependencyStructure gold = goldbank.get(i);
			Map<String, Integer> scores = computeScores(test, gold);
			label += scores.get("label");
			unlabel += scores.get("unlabel");
			total += test.items.length-1;
			if (scores.get("unlabel") == test.items.length - 1) 
				umatch++;
			if (scores.get("label") == test.items.length - 1) 
				lmatch++;
		}
		
		ret.put("LAS", label * 1./ total);
		ret.put("UAS", unlabel * 1./ total);
		ret.put("UEM", umatch * 1./ testbank.size());
		ret.put("LEM", lmatch * 1./testbank.size());
		return ret;
	}	
	
	public static void main(String[] args) throws IOException {
		
		String dic_path = "data/universal/dic/"; //args[0];
		String data_path = "data/universal/data/"; //args[1];
		String train_file = "train-small.conll";
		String kbest_file = "dev-300best-mst2ndorder.conll";
		String gold_file = "dev.conll";
		double alpha = 0.1;
		int K = 10;
		
		if (args.length == 7) {
			dic_path = args[0];
			data_path = args[1];
			train_file = args[2];
			kbest_file = args[3];
			gold_file = args[4];
			alpha = Double.valueOf(args[5]);
			K = Integer.valueOf(args[6]);
		}
		else if (args.length > 0) {
			System.err.println("dic_path data_path train_file kbest_file gold_file alpha K");
			return;
		}
		
		// load dics
		Lookup vocaDic = new Lookup(new Lookup.CollobertNormalizer()); 
		vocaDic.loadFromFile(dic_path + "/words.lst");
		if (!vocaDic.str2id.containsKey("UNKNOWN"))
			throw new Error ("not contain UNKNOWN");
		if (!vocaDic.str2id.containsKey("EOS")) 
			throw new Error("not contain EOS");

		Lookup posDic = new Lookup(new Lookup.NoNormalizer()); 
		posDic.loadFromFile(dic_path + "/pos.lst");
		if (!posDic.str2id.containsKey("EOS")) 
			throw new Error("not contain EOS");
		if (!posDic.str2id.containsKey("EOC")) 
			throw new Error("not contain EOC");
		
		Lookup deprelDic = new Lookup(new Lookup.NoNormalizer()); 
		deprelDic.loadFromFile(dic_path + "/deprel.lst");
		if (!deprelDic.str2id.containsKey("EOC")) 
			throw new Error("not contain EOC");
		
		Reranker reranker = new Reranker(vocaDic, posDic, deprelDic);
		
		// train
		reranker.train(data_path + train_file);
		
		// evaluate
		reranker.evaluate(data_path + kbest_file, data_path + gold_file, alpha, K);
	}
}
