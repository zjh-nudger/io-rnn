
public class DependencyItem {
	
	public int word, cap, tag, deprel;
	public int position;
	public int headPosition;
	public DependencyItem[] leftDeps, rightDeps;
	
	public DependencyItem() {
		this.word = -1;
		this.cap = -1;
		this.tag = -1;
		this.position = -1;
		this.headPosition = -1;
		this.deprel = -1;		
	}
	
	public DependencyItem(int token, int cap, int posTag, 
			int position, int headPosition, int depRelation) {
		this.word = token;
		this.cap = cap;
		this.tag = posTag;
		this.position = position;
		this.headPosition = headPosition;
		this.deprel = depRelation;
	}
	
	@Override
	public String toString() {
		return String.valueOf(word);
	}
	
	public String toString(Lookup vocaDic, Lookup posDic, Lookup deprelDic) {
		return 	String.valueOf(position) + "\t" + 
				"[" + String.valueOf(cap) + "] " + vocaDic.id2str.get(word) + "\t" + 
				posDic.id2str.get(tag) + "\t" + 
				String.valueOf(headPosition) + "\t" + 
				deprelDic.id2str.get(deprel);
	}
	
	@Override
	public int hashCode() {
		return word * 101 + tag * 31 + headPosition * 17 +  deprel;  
	}
	
	@Override
	public boolean equals(Object o) {
		if (o instanceof DependencyItem) {
			DependencyItem that = (DependencyItem)o;
			return word == that.word && tag == that.tag && position == that.position && 
					headPosition == that.headPosition && deprel == that.deprel;
		}
		else
			return false;
	}
}
