VER=SEQ
#VER=OpenMP
#VER=MPI
#VER=ComplexClass

#Sequential version
ifeq ($(VER), SEQ)
    EXE = gppKerSeq.ex
    SRC = gppKerSeq.cpp 
endif

#OpenMP3.5 version
ifeq ($(VER), OpenMP)
    EXE = gppOpenMP3.ex
    SRC = gppOpenMP3.cpp 
endif

#MPI version
ifeq ($(VER), MPI)
    EXE = gppMPI.ex
    SRC = gppMPIOpenMP3.cpp 
endif

#Complex class + gpp version
ifeq ($(VER), ComplexClass)
    EXE = gppComplex.ex
    SRC = gppComplex.cpp 
endif

CXX = CC

LINK = ${CXX}

ifeq ($(CXX),CC)
#cray flags
    CXXFLAGS=-O2 -hlist=a

#intel flags
#	CXXFLAGS=-O3 -qopenmp -std=c++11 -qopt-report=5
#	CXXFLAGS+=-xCORE_AVX2
##	CXXFLAGS+=-xMIC_AVX512
#	LINKFLAGS=-qopenmp -std=c++11
endif 

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ)  
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ1): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f $(OBJ) $(EXE)

