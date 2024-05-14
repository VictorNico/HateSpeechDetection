#!/bin/bash

cd frontend2
npm install
npm run build

cd ..
rm -rf backend/static
mv frontend2/dist/* backend/static
#mv modeling/dist/model.dat backend/model/

cd backend
pip install -r requirements.txt