uea_testing() {
python -u train.py ERing UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py Libras UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py AtrialFibrillation UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py BasicMotions UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py RacketSports UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py Handwriting UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py Epilepsy UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py JapaneseVowels UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py UWaveGestureLibrary UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py PenDigits UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py StandWalkJump UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py NATOPS UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py ArticularyWordRecognition UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py FingerMovements UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py LSST UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py HandMovementDirection UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py Cricket UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py CharacterTrajectories UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py EthanolConcentration UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py SelfRegulationSCP1 UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py SelfRegulationSCP2 UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py Heartbeat UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py PhonemeSpectra UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py SpokenArabicDigits UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py EigenWorms UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py DuckDuckGeese UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py PEMS-SF UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py FaceDetection UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py MotorImagery UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py InsectWingbeat UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval

}

seed=$1
rm training/uea/res_$seed.txt
uea_testing $seed 2>&1 | tee -a training/uea/res_$seed.txt