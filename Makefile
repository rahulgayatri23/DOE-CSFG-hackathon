#EXE = gppKerSeq.ex
#SRC = gppKerSeq.cpp 
#EXE = gppOpenMP3.ex
#SRC = gppOpenMP3.cpp 
#EXE = gppMPIOpenMP3.ex
#SRC = gppMPIOpenMP3.cpp 
EXE = gppComplex.ex
SRC = gppComplex.cpp 

CXX = CC

LINK = ${CXX}

ifeq ($(CXX),CC)
#cray flags
#    CXXFLAGS=-O2 -hlist=a

#intel 
	CXXFLAGS=-O3 -qopenmp -std=c++11 -qopt-report=5
	CXXFLAGS+=-xCORE_AVX2
#	CXXFLAGS+=-xMIC_AVX512
	LINKFLAGS=-qopenmp -std=c++11
endif 

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ)  
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ1): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f $(OBJ) $(EXE)

