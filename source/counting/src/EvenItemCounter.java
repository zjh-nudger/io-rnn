import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


public class EvenItemCounter {
	public Map<EventItem, Integer> count;
	
	public EvenItemCounter() {
		count = new HashMap<EventItem, Integer>();
	}
	
	public void put(EventItem item) {
		Integer c = count.get(item);
		if (c != null)
			count.put(item, c+1);
		else
			count.put(item, 1);
	}
	
	public int get(EventItem event) {
		Integer c = count.get(event);
		if (c == null) return 0;
		return c;
	}
	
	@Override 
	public String toString() {
		StringBuffer buff = new StringBuffer();
		for (Map.Entry<EventItem, Integer> entry : count.entrySet()) {
			buff.append("count: ").append(entry.getValue()).append("; ")
				.append("type : ").append(entry.getKey().type).append("; ")
				.append("item : ").append(Arrays.toString(entry.getKey().items)).append("; ")
				.append("\n");
		}
		return buff.toString();
	}
}
