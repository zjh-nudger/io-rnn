import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


public class Lookup {
	public Map<String, Integer> str2id;
	public Map<Integer, String> id2str;
	Normalizer norm;
	
	public static class Normalizer {
		public String normalize(String input) {
			return input;
		}
		public int getCapFeat(String input) {
			String lower = input.toLowerCase();
			String upper = input.toUpperCase();
			
			if (input.equals(lower))
				return ALL_LOWER;
			else if (input.equals(upper))
				return ALL_UPPER;
			else if (input.charAt(0) == upper.charAt(0))
				return FIRST_UPPER;
			else 
				return NOT_FIRST_UPPER;
		}
		
		public static int ALL_LOWER = 1;
		public static int ALL_UPPER = 2;
		public static int FIRST_UPPER = 3;
		public static int NOT_FIRST_UPPER = 4;

	}
	
	public static class NoNormalizer extends Normalizer {
	}
	
	public static class CollobertNormalizer extends Normalizer {
		@Override
		public String normalize(String word) {
			if (word.equals("UNKNOWN") || word.equals("EOS") || word.equals("<num>")) 
				return word;
			else if (word.equals("-LRB-")) return "(";
			else if (word.equals("-RRB-")) return ")";
			else if (word.equals("-LSB-")) return "[";
			else if (word.equals("-RSB-")) return "]";
			else if (word.equals("-LCB-")) return "{";
			else if (word.equals("-RCB-")) return "}";
			else
				return word.toLowerCase().replaceAll("^[0-9.,]+$", "<num>");
		}
	}
	
	public Lookup(Normalizer norm) {
		str2id = new HashMap<String, Integer>();
		id2str = new HashMap<Integer, String>();
		this.norm = norm;
	}
	
	public void loadFromFile(String path) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(path));
		int curId = 0;
		
		while (true) {
			String line = reader.readLine();
			if (line == null) break;
			
			line = line.trim();
			if (str2id.containsKey(line)) continue;
			str2id.put(line, curId);
			id2str.put(curId, line);
			curId++;
		}
		
		reader.close();
	}
	
	public int getId(String str) {
		Integer id = str2id.get(norm.normalize(str));
		if (id == null) {
			//System.out.println(str);
			return str2id.get("UNKNOWN");
		}
		else 
			return id;
	}	

	public static void main(String[] args) {
		String x = "12,45.99-abc";
		Normalizer norm = new CollobertNormalizer();
		System.out.println(norm.normalize(x));
	}
}
