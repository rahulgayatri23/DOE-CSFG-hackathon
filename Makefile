#EXE = gppKerSeq.ex
#SRC = gppKerSeq.cpp 
EXE = gppOpenMP3.ex
SRC = gppOpenMP3.cpp 

CXX = CC

LINK = ${CXX}

ifeq ($(CXX),CC)
#cray flags
    CXXFLAGS=-O2 -hlist=a
endif 

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ)  
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ1): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f $(OBJ) $(EXE)

