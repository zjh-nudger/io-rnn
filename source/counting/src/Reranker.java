import java.io.IOException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class Reranker {
	EvenItemCounter evCounter;
	Lookup vocaDic, posDic, deprelDic;
	
	public Reranker(Lookup vocaDic, Lookup posDic, Lookup deprelDic) {
		evCounter = new EvenItemCounter();
		this.vocaDic = vocaDic;
		this.posDic = posDic;
		this.deprelDic = deprelDic;
	}
	
	public void train(String dsbankpath) throws IOException {
		System.out.println("=============== training ============");
		System.out.println("loading file " + dsbankpath);
		CoNLLFormatReader reader = new CoNLLFormatReader(vocaDic, posDic, deprelDic);
		List<DependencyStructure> dsbank = reader.loadDSBank(dsbankpath);
		
		int i = 0;
		for (DependencyStructure ds : dsbank) {
			train(ds);
			
			if (i % 1000 == 0) System.out.print(".");
			i++;			
		}
		System.out.println();
	}
	
	public static enum EventType {
		DR_CONDEVENT	,	DR_EVENT, 
		TAG_CONDEVENT	, 	TAG_EVENT, 
		WORD_CONDEVENT	,	WORD_EVENT, 
		CAP_CONDEVENT	, 	CAP_EVENT, 
		DIST_CONDEVENT	, 	DIST_EVENT
	}
	
	
	public Map<EventType, EventItem[]> extractEvent(DependencyStructure ds, int headPosition,
			int depOrder, int dir) {
		Map<EventType, EventItem[]> ret = new EnumMap<EventType, EventItem[]>(EventType.class);

		DependencyItem head	 = ds.items[headPosition];
		DependencyItem grand = new DependencyItem();
		if (head.position > 0)
			grand = ds.items[head.headPosition];

		DependencyItem dep;
		DependencyItem sis;
		DependencyItem[] deps;
		if (dir == 0) 	deps = head.leftDeps;
		else			deps = head.rightDeps;
		if (depOrder < deps.length)
			dep = deps[depOrder];
		else 
			dep = new DependencyItem(-1, -1, posDic.getId("EOC"), -1, headPosition, deprelDic.getId("EOC"));
		if (depOrder > 0)
			sis = deps[depOrder-1];
		else
			sis = new DependencyItem();
						
		int dist = Math.abs(dep.position - head.position);
		if 		(dist == 1) dist = 1;
		else if (dist == 2) dist = 2;
		else if (dist <= 6) dist = 3;
		else 				dist = 4;
	
		// for P(deprel(D) | H,S,G, dir)
		// conditional event; reduction list
		ret.put(EventType.DR_CONDEVENT, new EventItem[] {
				new EventItem(ret.size(), new int[] {head.word, head.tag, head.deprel, sis.word, sis.tag, grand.word, grand.tag, dir}),
				new EventItem(ret.size(), new int[] {head.word, head.tag, head.deprel, sis.word, sis.tag, grand.tag, dir}),
				new EventItem(ret.size(), new int[] {head.word, head.tag, head.deprel, sis.tag, grand.tag, dir}),	//*
				new EventItem(ret.size(), new int[] {head.tag, head.deprel, sis.word, sis.tag, grand.tag, dir}),		//*
				new EventItem(ret.size(), new int[] {head.tag, head.deprel, sis.tag, grand.tag, dir})
			});

		// event
		ret.put(EventType.DR_EVENT, new EventItem[] {
				new EventItem(ret.size(), new int[] {dep.deprel, head.word, head.tag, head.deprel, sis.word, sis.tag, grand.word, grand.tag, dir}),
				new EventItem(ret.size(), new int[] {dep.deprel, head.word, head.tag, head.deprel, sis.word, sis.tag, grand.tag, dir}),
				new EventItem(ret.size(), new int[] {dep.deprel, head.word, head.tag, head.deprel, sis.tag, grand.tag, dir}),	//*
				new EventItem(ret.size(), new int[] {dep.deprel, head.tag, head.deprel, sis.word, sis.tag, grand.tag, dir}),		//*
				new EventItem(ret.size(), new int[] {dep.deprel, head.tag, head.deprel, sis.tag, grand.tag, dir})
			});

		// if EOC, return
		if (dep.deprel == deprelDic.getId("EOC")) return ret;

		// for P(tag(D) | H,S,G, dir)
		// conditional event; reduction list
		ret.put(EventType.TAG_CONDEVENT, new EventItem[] {
				new EventItem(ret.size(), new int[] {dep.deprel, head.word, head.tag, head.deprel, sis.tag,  dir}),
				new EventItem(ret.size(), new int[] {dep.deprel, head.deprel, sis.tag, dir}),
				new EventItem(ret.size(), new int[] {dep.deprel, head.deprel,  dir})
			});

		// event
		ret.put(EventType.TAG_EVENT, new EventItem[] {
				new EventItem(ret.size(), new int[] {dep.tag, dep.deprel, head.word, head.tag, head.deprel, sis.tag,  dir}),
				new EventItem(ret.size(), new int[] {dep.tag, dep.deprel, head.deprel, sis.tag, dir}),
				new EventItem(ret.size(), new int[] {dep.tag, dep.deprel, head.deprel,  dir})
		});
		
		
		// for P(word(D) | tag(D), H,S,G, dir)
		// conditional event; reduction list
		ret.put(EventType.WORD_CONDEVENT, new EventItem[] {
				new EventItem(ret.size(), new int[] {head.word, head.tag, head.deprel, sis.tag, dir}),
				new EventItem(ret.size(), new int[] {head.tag, head.deprel, sis.tag, dir})
			});
				
		// event
		ret.put(EventType.WORD_EVENT, new EventItem[] {
				new EventItem(ret.size(), new int[] {dep.word, head.word, head.tag, head.deprel, sis.tag, dir}),
				new EventItem(ret.size(), new int[] {dep.word, head.tag, head.deprel, sis.tag, dir})
			});
	
		// for P(cap(D) | word(D), tag(D), H,S,G, dir)
		// conditional event; reduction list
		ret.put(EventType.CAP_CONDEVENT, new EventItem[] {
				new EventItem(ret.size(), new int[] {dep.word, dep.tag,  head.deprel, dir}),
				new EventItem(ret.size(), new int[] {dep.word, dep.tag, dir})
			});
				
		// event
		ret.put(EventType.CAP_EVENT, new EventItem[] {
				new EventItem(ret.size(), new int[] {dep.cap, dep.word, dep.tag,  head.deprel, dir}),
				new EventItem(ret.size(), new int[] {dep.cap, dep.word, dep.tag, dir})
			});
	
		// for P(dist(H,D) | word(D), tag(D), H,S,G, dir)
		// conditional event; reduction list
		ret.put(EventType.DIST_CONDEVENT, new EventItem[] {
				new EventItem(ret.size(), new int[] {dep.word, dep.tag, dep.deprel, head.tag, head.deprel, sis.tag, dir}),
				new EventItem(ret.size(), new int[] {dep.tag, dep.deprel, head.tag, head.deprel, sis.tag, dir})
			});

		// event
		ret.put(EventType.DIST_EVENT, new EventItem[] {
				new EventItem(ret.size(), new int[] {dist, dep.word, dep.tag, dep.deprel, head.tag, head.deprel, sis.tag, dir}),
				new EventItem(ret.size(), new int[] {dist, dep.tag, dep.deprel, head.tag, head.deprel, sis.tag, dir})
			});
		
		return ret;
	}
	
	public void train(DependencyStructure ds) {
		for (int i = 0; i < ds.items.length; i++) {
			DependencyItem item = ds.items[i];
			
			// leftdeps
			for (int j = 0; j <= item.leftDeps.length; j++) {
				Map<EventType, EventItem[]> events = this.extractEvent(ds, i, j, 0);
				for (EventItem[] its : events.values()) {
					for (EventItem it : its)
						evCounter.put(it);
				}
			}
			
			// rightdeps
			for (int j = 0; j <= item.rightDeps.length; j++) {
				Map<EventType, EventItem[]> events = this.extractEvent(ds, i, j, 1);
				for (EventItem[] its : events.values()) {
					for (EventItem it : its)
						evCounter.put(it);
				}
			}
		}
	}
	
	public double computeLogProb(DependencyStructure ds) {
		double logProb = 0;
		for (int i = 0; i < ds.items.length; i++) {
			DependencyItem item = ds.items[i];
			
			// leftdeps
			for (int j = 0; j <= item.leftDeps.length; j++) {
				Map<EventType, EventItem[]> events = this.extractEvent(ds, i, j, 0);
				logProb += computeLogProb(events);
			}
			
			// rightdeps
			for (int j = 0; j <= item.rightDeps.length; j++) {
				Map<EventType, EventItem[]> events = this.extractEvent(ds, i, j, 1);
				logProb += computeLogProb(events);
			}
		}
		return logProb;
	}

	Map<EventType, Map<Integer,Integer>> countAppear = null;
	
	public double computeLogProb(Map<EventType, EventItem[]> events) {
		double logProb = 0;
		
		double p = 0;
		EventItem[] ev;
		EventItem[] cEv;

		// for statistics 
		if (countAppear == null)
			countAppear = new EnumMap<EventType, Map<Integer,Integer>>(EventType.class);

		
		// deprel
		p = 0;
		ev = events.get(EventType.DR_EVENT);
		cEv = events.get(EventType.DR_CONDEVENT);
		p = (evCounter.get(ev[4]) + 0.005) / (evCounter.get(cEv[4]) + 0.5);
		p = (evCounter.get(ev[3]) + evCounter.get(ev[2]) + 3*p) / 
			(evCounter.get(cEv[3]) + evCounter.get(cEv[2]) + 3);
		p = (evCounter.get(ev[1]) + 3*p) / (evCounter.get(cEv[1]) + 3);
		p = (evCounter.get(ev[0]) + 3*p) / (evCounter.get(cEv[0]) + 3);
		logProb += Math.log(p);

		Map<Integer,Integer> local = countAppear.get(EventType.DR_EVENT);
		if (local == null) {
			local = new HashMap<Integer,Integer>();
			countAppear.put(EventType.DR_EVENT, local);
			for (int i = 0; i < ev.length; i++) 
				local.put(i, 0);
		}
		for (int i = 0; i < ev.length; i++) {
			if (evCounter.get(ev[i]) > 2.5) 
				local.put(i, local.get(i) + 1);
		}

		if (events.get(EventType.TAG_EVENT) == null)
			return logProb; // EOC
	
		// tag
		p = 0;
		ev = events.get(EventType.TAG_EVENT);
		cEv = events.get(EventType.TAG_CONDEVENT);
		p = (evCounter.get(ev[2]) + 0.005) / (evCounter.get(cEv[2]) + 0.5);
		p = (evCounter.get(ev[1]) + 3*p) / (evCounter.get(cEv[1]) + 3);
		p = (evCounter.get(ev[0]) + 3*p) / (evCounter.get(cEv[0]) + 3);
		logProb += Math.log(p);

		local = countAppear.get(EventType.TAG_EVENT);
		if (local == null) {
			local = new HashMap<Integer,Integer>();
			countAppear.put(EventType.TAG_EVENT, local);
			for (int i = 0; i < ev.length; i++) 
				local.put(i, 0);
		}
		for (int i = 0; i < ev.length; i++) {
			if (evCounter.get(ev[i]) > 2.5) 
				local.put(i, local.get(i) + 1);
		}

	
		// word
		p = 0;
		ev = events.get(EventType.WORD_EVENT);
		cEv = events.get(EventType.WORD_CONDEVENT);
		p = (evCounter.get(ev[1]) + 0.005) / (evCounter.get(cEv[1]) + 0.5);
		p = (evCounter.get(ev[0]) + 3*p) / (evCounter.get(cEv[0]) + 3);
		logProb += Math.log(p);
	
		local = countAppear.get(EventType.WORD_EVENT);
		if (local == null) {
			local = new HashMap<Integer,Integer>();
			countAppear.put(EventType.WORD_EVENT, local);
			for (int i = 0; i < ev.length; i++) 
				local.put(i, 0);
		}
		for (int i = 0; i < ev.length; i++) {
			if (evCounter.get(ev[i]) > 2.5) 
				local.put(i, local.get(i) + 1);
		}

		// word
		p = 0;
		ev = events.get(EventType.CAP_EVENT);
		cEv = events.get(EventType.CAP_CONDEVENT);
		p = (evCounter.get(ev[1]) + 0.005) / (evCounter.get(cEv[1]) + 0.5);
		p = (evCounter.get(ev[0]) + 3*p) / (evCounter.get(cEv[0]) + 3);
		logProb += Math.log(p);

		local = countAppear.get(EventType.CAP_EVENT);
		if (local == null) {
			local = new HashMap<Integer,Integer>();
			countAppear.put(EventType.CAP_EVENT, local);
			for (int i = 0; i < ev.length; i++) 
				local.put(i, 0);
		}
		for (int i = 0; i < ev.length; i++) {
			if (evCounter.get(ev[i]) > 2.5) 
				local.put(i, local.get(i) + 1);
		}

		// dist
		// comment the following code if computing perplexity
		p = 0;
		ev = events.get(EventType.DIST_EVENT);
		cEv = events.get(EventType.DIST_CONDEVENT);
		p = (evCounter.get(ev[1]) + 0.005) / (evCounter.get(cEv[1]) + 0.5);
		p = (evCounter.get(ev[0]) + 3*p) / (evCounter.get(cEv[0]) + 3);
		//logProb += Math.log(p);

		return logProb;
	}
	
	public static double maximum(double[] xs) {
		double m = xs[0];
		for (int i = 1; i < xs.length; i++) {
			if (xs[i] > m) 
				m = xs[i];
		}
		return m;
	}
	
	public static double logSumOfExponentials(double[] xs) {
		if (xs.length == 1)
			return xs[0];
		double max = maximum(xs);
		double sum = 0.0;
		for (int i = 0; i < xs.length; ++i)
			if (xs[i] != Double.NEGATIVE_INFINITY)
				sum += java.lang.Math.exp(xs[i] - max);
		return max + java.lang.Math.log(sum);
	}
	
	public List<DependencyStructure> rerank(List<List<DependencyStructure>> kbestbank, List<double[]> mstLogProbs, 
									List<Double> ppl, String output, double alpha, int K) throws IOException {

		List<DependencyStructure> ret = new LinkedList<DependencyStructure>();
		double sum_sen_log_p = 0;
		int sum_n_words = 0;
		
		BufferedWriter writer = null;
		if (output != null) 
			writer = new BufferedWriter(new FileWriter(output));
		
		int i = 0;
		for (List<DependencyStructure> parses : kbestbank) {
			double higestLogProb = -1e10;
			DependencyStructure bestDs = null;
			double[] logProbs = new double[parses.size()];
			double[] mstLogP = mstLogProbs.get(i);

			int j = 0;
			for (DependencyStructure ds : parses) {
				ds.scores[0] = mstLogP[j];
				ds.scores[1] = computeLogProb(ds);
				double logProb = (1-alpha) * ds.scores[1] + alpha * ds.scores[0] ;
				logProbs[j] = logProb;
				
				if (bestDs == null || higestLogProb < logProb) {
					higestLogProb = logProb;
					bestDs = ds;
				}
				j++;
				if (j >= K) break;
			}			
			ret.add(bestDs);
			if (writer != null) 
				writer.write(bestDs.raw);

			sum_sen_log_p += logSumOfExponentials(logProbs);
			sum_n_words += parses.get(0).items.length - 1;
			
			if (i % 100 == 0) System.out.print(".");
			i++;
		}
		System.out.println();

		if (writer != null) writer.close();
		
		ppl.add(Math.pow(2, -sum_sen_log_p / Math.log(2) / sum_n_words));
		
		return ret;
	}

	public List<DependencyStructure> rerankWithOracle(List<List<DependencyStructure>> kbestbank, 
			List<DependencyStructure> goldbank, int K) {
		List<DependencyStructure> ret = new LinkedList<DependencyStructure>();
		
		int i = 0;
		for (List<DependencyStructure> parses : kbestbank) {
			double higestUnlabel = -1e10;
			DependencyStructure bestDs = null;
			int j = 0;
			for (DependencyStructure ds : parses) {
				double unlabel = computeScores(ds, goldbank.get(i)).get("unlabel"); 
				if (bestDs == null || higestUnlabel < unlabel) {
					higestUnlabel = unlabel;
					bestDs = ds;
				}
				j++;
				if (j >= K) break;
			}
			i++;
			ret.add(bestDs);
		}
		return ret;
	}

	public List<DependencyStructure> norerank(List<List<DependencyStructure>> kbestbank) {
		List<DependencyStructure> ret = new LinkedList<DependencyStructure>();
		for (List<DependencyStructure> parses : kbestbank) {
			ret.add(parses.get(0));
		}
		return ret;
	}
	
	public Map<String, Integer> computeScores(DependencyStructure test, DependencyStructure gold) {
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
	
	public Map<String, Double> computeScores(List<DependencyStructure> testbank, 
			List<DependencyStructure> goldbank) {
		if (testbank.size() != goldbank.size())
			throw new Error("size not match " + testbank.size() + " " + goldbank.size());
		
		Map<String, Double> ret = new HashMap<String, Double>();
		int label = 0;
		int unlabel = 0;
		int umatch = 0;
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
		}
		
		ret.put("LAS", label * 1./ total);
		ret.put("UAS", unlabel * 1./ total);
		ret.put("UEM", umatch * 1./ testbank.size());
		return ret;
	}
	
	public double computePerplexity(List<DependencyStructure> dsbank) {
		double perplex = 0;
		
		int i = 0;
		int nwords = 0;
		for (DependencyStructure ds : dsbank) {
			double logP = computeLogProb(ds) / Math.log(2); 
			perplex += logP;
			nwords += ds.items.length - 1;
			
			if (i % 100 == 0) System.out.print(".");
			i++;
		}
		return Math.pow(2, -perplex / nwords);
	}

    public List<double[]> loadMSTLogProbs(String filename) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		List<double[]> logProbs = new LinkedList<double[]>();

		while (true) {
			String line = reader.readLine();
			if (line == null) break;

			String[] comps = line.trim().split(" ");
			double[] scores = new double[comps.length];
			for (int i = 0; i < scores.length; i++) 
				scores[i] = Double.valueOf(comps[i]);
			double logSum = logSumOfExponentials(scores);
			for (int i = 0; i < scores.length; i++)
				scores[i] -= logSum;
			
			logProbs.add(scores);
		}
		reader.close();
		return logProbs;
	}

	public void evaluate(String kbestbankpath, String goldbankpath, String output, double alpha, int K) throws IOException {
		System.out.println("================== evaluation ============");
		System.out.println("loading files " + kbestbankpath + " ; " + goldbankpath);
		CoNLLFormatReader reader = new CoNLLFormatReader(vocaDic, posDic, deprelDic);
		List<double[]> mstLogProbs = loadMSTLogProbs(kbestbankpath+".mstscores");
		List<List<DependencyStructure>> kbestbank = reader.loadKbestDSBank(kbestbankpath, mstLogProbs); 
		List<DependencyStructure> goldbank = reader.loadDSBank(goldbankpath);
		
		// perplexity
		double perplexity = computePerplexity(goldbank);
		System.out.format("ds-ppl %.2f \n", perplexity);
				
		// w/o rerank
		List<DependencyStructure> norerankparses = norerank(kbestbank);
		Map<String, Double> norerankScores = computeScores(norerankparses, goldbank);
		System.out.format("w/o rerank: LAS %.2f ; UAS %.2f \n", norerankScores.get("LAS")*100, 
				norerankScores.get("UAS")*100);
		
		// oracle

		System.out.println("========== K " + K + " =========");
		List<DependencyStructure> oracleparses = rerankWithOracle(kbestbank, goldbank, K);
		Map<String, Double> oracleScores = computeScores(oracleparses, goldbank);
		System.out.format("oracle: LAS %.2f ; UAS %.2f \n", oracleScores.get("LAS")*100, 
				oracleScores.get("UAS")*100);

		// rerank
		List<Double> ppl = new LinkedList<Double>();
		// with alpha = 0
		List<DependencyStructure> rerankparses = rerank(kbestbank, mstLogProbs, ppl, output, alpha, K);
		Map<String, Double> rerankScores = computeScores(rerankparses, goldbank);
		System.out.format("rerank :  LAS %.2f ; UAS %.2f \n",
									rerankScores.get("LAS")*100, 
									rerankScores.get("UAS")*100);
	}
	
	public static void main(String[] args) throws IOException {
		
		String dic_path = "../data/wsj10/dic/"; //args[0];
		String data_path = "../data/wsj10/data/"; //args[1];
		String train_file = "train.conll";
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
		reranker.evaluate(data_path + kbest_file, data_path + gold_file, "x", alpha, K);
	}
}
