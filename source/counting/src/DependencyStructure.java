import java.util.Collections;
import java.util.LinkedList;
import java.util.List;


public class DependencyStructure {
	public DependencyItem[] items;
	public double[] scores;
	public String raw;
	
	public DependencyStructure(DependencyItem[] items, String raw, double[] scores) {
		this.items = items;
		this.raw = raw;
		this.scores = scores;
		
		for (DependencyItem item : items) {
			List<DependencyItem> leftDeps = new LinkedList<DependencyItem>();
			List<DependencyItem> rightDeps = new LinkedList<DependencyItem>();
			
			for (DependencyItem item1 : items) {
				if (item1.headPosition == item.position) {
					if (item1.position < item.position)
						leftDeps.add(item1);
					else 
						rightDeps.add(item1);
				}
			}
			
			Collections.reverse(leftDeps);
			item.leftDeps = leftDeps.toArray(new DependencyItem[0]);
			item.rightDeps = rightDeps.toArray(new DependencyItem[0]);
		}
	}

	public boolean sameSentence(DependencyStructure that) {
		if (items.length != that.items.length) 
			return false;
		for (int i = 0; i < items.length; i++) {
			if (items[i].word != that.items[i].word)
				return false;
		}
		return true;
	}
	
	public String toString(Lookup vocaDic, Lookup posDic, Lookup deprelDic) {
		StringBuffer buff = new StringBuffer();
		for (DependencyItem dep : items) {
			buff.append(dep.toString(vocaDic, posDic, deprelDic)).append("\n");
		}
		return buff.toString();
		
	}	
}
