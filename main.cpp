#include <pistache/endpoint.h>
#include <pistache/http.h>
#include <pistache/router.h>
#include <string>
#include <vector>

#include <torch/script.h>
#include <iostream>

const int PORT = 9900;
const std::string MODEL_PATH = "./model/lstm_jit_traced.pt";

using namespace Pistache;

std::vector<float> moving_average(std::vector<float> arr, int windowSize = 3)
{
  float sum = 0.0;
  float movingAverage = 0.0;
  std::vector<float> moving_avg;

  for (int i = 0; i < (arr.size() - windowSize); i++)
  {
    sum = 0.0; // Reinitialize sum back to zero.
    for (int j = i; j < i + windowSize; j++)
    {
      sum += arr[j];
    }
    moving_avg.push_back(sum / windowSize);
  }

  return moving_avg;
}

float getPointFromModel(std::vector<float> points)
{
  torch::jit::script::Module module;
  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(MODEL_PATH);
  }
  catch (const c10::Error &e)
  {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::vector<float> movingAvg = moving_average(points, 3);
  // std::cout<<"moving avg:  " << movingAvg<<"\n";
  auto options = torch::TensorOptions().dtype(at::kFloat);
  torch::Tensor inputs_tensor = torch::from_blob(movingAvg.data(), {1, (int)movingAvg.size()}, options).clone();
  auto shaped_tensor = inputs_tensor.view({1, 12, 1});
  std::vector<torch::jit::IValue> inputs_values;

  inputs_values.push_back(shaped_tensor);

  // std::cout<<"tensor shape" << unsqueezed_tensor.sizes();
  at::Tensor output = module.forward(inputs_values).toTensor();
  float res_arr[1] = {output[0][0].item<float>()};
  int length = points.size();
  float lastPoint = res_arr[0];
  if(length > 1) lastPoint = (res_arr[0] * 3) - (points[length - 1] + points[length - 2]);
  return lastPoint;
}

std::vector<float> splitPoints(std::string pointsString)
{
  std::vector<float> pointsFloat;
  char *char_array = new char[pointsString.size() + 1];
  strcpy(char_array, pointsString.c_str());
  char *p;
  p = strtok(char_array, ",");
  while (p != NULL)
  {
    pointsFloat.push_back(stof((std::string)(p)));
    p = strtok(NULL, ",");
  }
  return pointsFloat;
}

void getForecastingData(const Rest::Request &req, Http::ResponseWriter resp)
{
  // get the parameter value, default to an warning message if it's not available
  std::string text = req.hasParam(":text") ? // check if parameter is included
                         req.param(":text").as<std::string>()
                                           :       // if so set as text value
                         "No parameter supplied."; // otherwise return warning
  std::cout<<text;
  std::vector<float> points = splitPoints(text);
  for (float point : points)
  {
    std::cout << point << std::endl;
  }
  float point = getPointFromModel(points);
  std::cout<<point;
  resp.send(Http::Code::Ok, std::to_string(point)); // return a response from our server
}

int main(int argc, char *argv[])
{
  using namespace Rest;

  Router router;   // POST/GET/etc. route handler
  Port port(PORT); // port to listen on
  Address addr(Ipv4::any(), port);
  std::shared_ptr<Http::Endpoint> endpoint = std::make_shared<Http::Endpoint>(addr);
  auto opts = Http::Endpoint::options().threads(1); // how many threads for the server
  endpoint->init(opts);

  /* routes! */
  std::cout<<"start listening";
  Routes::Get(router, "/forecasting/:text", Routes::bind(&getForecastingData));
  endpoint->setHandler(router.handler());
  endpoint->serve();
  return 0;
}