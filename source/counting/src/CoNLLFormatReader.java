import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;


public class CoNLLFormatReader {
	
	Lookup vocaDic, posDic, deprelDic;

	public CoNLLFormatReader(Lookup vocaDic, Lookup posDic, Lookup deprelDic) {
		this.vocaDic = vocaDic;
		this.posDic = posDic;
		this.deprelDic = deprelDic;
	}
	
	public List<List<DependencyStructure>> loadKbestDSBank(String path, List<double[]> scores) throws IOException {
		List<List<DependencyStructure>> ret = new LinkedList<List<DependencyStructure>>();
		List<DependencyStructure> bank = this.loadDSBank(path);
		
		int n = 0;
		for (double[] s : scores) {
			List<DependencyStructure> kbest = bank.subList(n, n+s.length);
			ret.add(kbest);
			n += s.length;
		}
		
		return ret;
	}
	
	public List<DependencyStructure> loadDSBank(String path) throws IOException {
		List<DependencyStructure> ret = new LinkedList<DependencyStructure>();
		BufferedReader reader = new BufferedReader(new FileReader(path));
		List<DependencyItem> curItems = new LinkedList<DependencyItem>();
		curItems.add(new DependencyItem(vocaDic.getId("EOS"), -1,
										posDic.getId("EOS"),
										0, -1, -1));
		
		StringBuffer raw = new StringBuffer();

		while (true) {
			String line = reader.readLine();
			if (line == null) break;
			line = line.trim();
			raw.append(line).append("\n");
			
			// if this line empty, turn to new dep. structure
			if (line.isEmpty()) {
				ret.add(new DependencyStructure(curItems.toArray(new DependencyItem[0]), raw.toString(), new double[]{0,0}));

				curItems.clear();
				raw = new StringBuffer();
				
				// create ROOT
				try {
					curItems.add(new DependencyItem(vocaDic.getId("EOS"), -1,
													posDic.getId("EOS"),
													0, -1, -1));
				}
				catch(Exception e) {
					System.err.println("lookups not contain EOS");
				}
				
				continue;
			}
			
			// else 
			String[] comps = line.split("\t"); //System.out.println(buff);
			int position = Integer.parseInt(comps[0]);
			String tok = comps[1];
			String posTag = comps[4];
			int headPosition = Integer.parseInt(comps[6]);
			String depRel = comps[7];
			DependencyItem item = new DependencyItem(	vocaDic.getId(tok), vocaDic.norm.getCapFeat(tok),
														posDic.getId(posTag),
														position, 
														headPosition, 
														deprelDic.getId(depRel));
			curItems.add(item);
		}
		
		reader.close();
		return ret;
	}
	
	public static void main(String[] args) throws IOException {
		
		String dic_path = "./toy/dic/"; //args[0];
		String data_path = "./toy/data/"; //args[1];
		
		// load dics
		Lookup vocaDic = new Lookup(new Lookup.CollobertNormalizer()); 
		vocaDic.loadFromFile(dic_path + "/words.lst");
		if (!vocaDic.str2id.containsKey("UNKNOWN")) 
			throw new Error ("not contain UNKNOWN");
		if (!vocaDic.str2id.containsKey("EOS")) 
			throw new Error("not contain EOS");

		Lookup posDic = new Lookup(new Lookup.NoNormalizer()); 
		posDic.loadFromFile(dic_path + "/cpos.lst");
		if (!posDic.str2id.containsKey("EOS")) 
			throw new Error("not contain EOS");
		
		Lookup deprelDic = new Lookup(new Lookup.NoNormalizer()); 
		deprelDic.loadFromFile(dic_path + "/deprel.lst");
		if (!deprelDic.str2id.containsKey("EOC")) 
			throw new Error("not contain EOC");
		
		CoNLLFormatReader reader = new CoNLLFormatReader(vocaDic, posDic, deprelDic);
		List<DependencyStructure> dsbank = reader.loadDSBank(data_path + "/train.conll");
		
		for (DependencyItem dep : dsbank.get(0).items) {

			System.out.println(dep.toString(vocaDic, posDic, deprelDic));
		}
	}
}
