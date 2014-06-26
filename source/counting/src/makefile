JFLAGS = -cp "./commons-exec-1.2/commons-exec-1.2.jar:./" -g
JC = javac
.SUFFIXES: .java .class
.java.class:
		$(JC) $(JFLAGS) $*.java

CLASSES = \
		Lookup.java \
		DependencyItem.java  \
		DependencyStructure.java \
		EvenItemCounter.java \
		EventItem.java \
		Pair.java \
		CoNLLFormatReader.java  \
		Reranker.java \
		UnsupParser.java

default: classes

classes: $(CLASSES:.java=.class)

clean:
		$(RM) *.class


















