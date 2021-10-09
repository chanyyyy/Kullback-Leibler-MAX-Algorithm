def KL_MAX(kernel,C,gamma,X_V,Y_V,X_U,Y_U,X_Test,Y_Test,iters,batch_size,path)
  model_1 = SVC(kernel=kernel, C=C, gamma=gamma, random_state=1,probability=True)
  model_2 = SVC(kernel=kernel, C=C, gamma=gamma, random_state=1,probability=True)
  model_1.fit(X_V,Y_V)
  lent = X_U.shape[0]-1
  KLM = []
  for i in range(lent+1):
      target = X_U[i].reshape(1, -1)
      X_noTar = X_U.copy()
      X_noTar = np.delete(X_noTar,i,axis=0)
      PLPP = model_1.predict_proba(target)
      X_L = np.vstack((X_V,X_U[i]))
      Y_L = np.concatenate((Y_V,[Y_U[i]]))
      model_2.fit(X_L,Y_L)
      PL = model_1.predict_proba(X_noTar)
      PLP = model_2.predict_proba(X_noTar)
      KL = scipy.stats.entropy(PLP,PL)
      k = sum(KL*PLPP/lent)
      KLM.append(sum(k))
  KLM = np.array(KLM)
  np.savetxt('KLM.csv',KLM)
  KLM = np.sum(KLM,axis=1)
  model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=1,probability=True)
  for count in range(iters):
      model.fit(X_V, Y_V)
      y_pred = model.predict(X_Test)
      accuracy.append(accuracy_score(y_pred,Y_Test))
      if accuracy[count]>accumax:
          accumax = accuracy[count]
          joblib.dump(model,path)
      ind = topk_partition(KLM,batch_size) # Sort and get a specified number of indexes
      X_V = np.concatenate((X_V,X_U[ind]))
      Y_V = np.concatenate((Y_V,Y_U[ind]))
      X_U = np.delete(X_U,ind,0)
      Y_U = np.delete(Y_U,ind,0)
  accuracy = np.array(accuracy)
