
public class EventItem {
	int[] items;
	int type; 
	
	public EventItem(int type, int[] items) {
		this.items = items;
		this.type = type;
	}
	
	@Override
	public int hashCode() {
		int hash = 1;
		for (int i = 0; i < items.length; i++)
			hash = hash * 31 + items[i];
		hash = hash * 31 + items.length * 11 + type;
		return hash;
	}
	
	@Override
	public boolean equals(Object o) {
		if (o instanceof EventItem) {
			EventItem that = (EventItem)o;
			if (type != that.type || items.length != that.items.length)
				return false;
			for (int i = 0; i < items.length; i++) {
				if (items[i] != that.items[i])
					return false;
			}
			return true;
		}
		else 
			return false;
	}
}
