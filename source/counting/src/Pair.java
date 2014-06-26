
public class Pair<T1, T2> {
	T1 first;
	T2 second;
	
	public Pair(T1 first, T2 second){
		this.first = first;
		this.second = second;
	}
	
	public T1 getFirst() {
		return first;
	}
	
	public T2 getSecond() {
		return second;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public boolean equals( Object other) {
		if (other instanceof Pair ) {
			Pair<T1,T2> that = (Pair<T1, T2>)other;
			return (this.first.equals(that.first) && this.second.equals(that.second));
		}
		else return false;
	}
	
	@Override
	public int hashCode() {
		return first.hashCode() * 31 + second.hashCode();
	}
}
