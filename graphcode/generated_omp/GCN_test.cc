#include"GCN_test.h"

void GCN(graph& g , vector<int> neuronsPerHiddenLayer)
{
   GNN gnn(g, "../sample_graphs/Amazon/amazon_features.txt", "../sample_graphs/Amazon/amazon_labels.txt");
  gnn.gcnPreprocessing();
  gnn.initializeLayers(neuronsPerHiddenLayer,"he");

  int num_epoch = 0;
  bool x = true;
  
	clock_t time_req;

	// Using pow function
	time_req = clock();
    while (x){
    if (num_epoch == 100 )
      {
      x = false;
    }
    num_epoch = num_epoch + 1;
    int layerr = 0;
    bool y = true;
    while (y){
      layerr = layerr + 1;
      #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic)
      for (int nod = 0; nod < g.num_nodes(); nod ++) 
      {
        if (layerr == neuronsPerHiddenLayer.size() + 1 )
          {
          y = false;
        }
        gnn.forwardPass(nod,layerr,1);

      }
    }
    while (layerr){
        gnn.backPropogation(layerr);
        layerr--;
    }
    gnn.adamOptimizer(num_epoch,0.01,0.9,0.999,1e-8);
    cout << "Epoch number is : "<<num_epoch << " with " ;
    gnn.predict();
  }
time_req = clock()- time_req;
cout << "Time taken for the execution is : " << (float)time_req/CLOCKS_PER_SEC << " seconds" << endl;
}
int main(){
    graph G("../sample_graphs/Amazon/amazon_edgelist.txt");

  G.parseGraph();
  vector<int> neuronsPerHiddenLayer;
  neuronsPerHiddenLayer.push_back(16);
  GCN(G,neuronsPerHiddenLayer);
  return 0;

}

