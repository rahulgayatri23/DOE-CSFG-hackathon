# DOE-CSGF

What are the different versions of the files :
1) gppKerSeq.cpp - Unoptimized sequential version
2) gppOpenMP3.cpp - OpenMP3.5 version
3) gppMPIOpenMP3.cpp - OpenMP3.5 version
4) gppComplex.cpp - The most optimized version which uses a Complex class for its computations.

Makefile - Modify based on version one is working with.
testSmallJob.pbs - a small test job for titan.
testMedJob.pbs - a medium test job for titan.
testBigJob.pbs - a Big test job for titan.

Submit Job command - qsub job.pbs
    Returns a JobId

Request interactive node -
    qsub -I -A CSC261 -q debug -l nodes=1,walltime=60:0

Check Job Status - showq | grep JobId
Results will get stored
    Gpp{*}Job.node_id


Results expected
    Small Problem Size :
        achtemp[0] = (-2.33321481494884e+17,-2.30534726978554e+17)
        achtemp[1] = (-2.34252423351125e+17,-2.30513076345906e+17)
        achtemp[2] = (-2.35188613227291e+17,-2.30484574707011e+17)
        ********** Time Taken **********= 14.284598 secs

    Med Problem Size :
    achtemp[0] = (-1.00387366710739e+19,-9.9815856421254e+18)
    achtemp[1] = (-1.00578107032611e+19,-9.98139840041474e+18)
    achtemp[2] = (-1.00769357238471e+19,-9.98115517772354e+18)
    ********** Time Taken **********= 57.107071 secs

    Big Problem Size :
        achtemp[0] = (-5.13842445834172e+20,-5.12423661308937e+20)
        achtemp[1] = (-5.14315659139572e+20,-5.12421580271217e+20)
        achtemp[2] = (-5.14789457600568e+20,-5.12418893702435e+20)
        ********** Time Taken **********= 228.427529 secs

