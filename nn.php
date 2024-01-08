<?php

namespace nn;

#####################################################################################################

# https://github.com/CharlesFr/ANN_Tutorial
# https://victorzhou.com/blog/intro-to-neural-networks/

#####################################################################################################

function settings()
{
  global $settings;
  \webdb\utils\simple_app_settings();
  $settings["application_default_page_id"]="nn";
  $settings["app_name"]="nn";
  $settings["app_title"]="nn";
  $settings["email_link_prefix"]="";
  $settings["dev_env"]=true;
  $settings["dev_env_email"]="";
  $settings["email_file_log_enabled"]=false;
  if (isset($settings["app_root_path"])==false)
  {
    $settings["app_root_path"]=$settings["env_root_path"];
  }
  $settings["app_directory_name"]=basename($settings["app_root_path"]);
  $settings["app_web_root"]="/".$settings["app_directory_name"]."/";
  $settings["app_web_resources"]=$settings["app_web_root"]."resources/";
  $settings["app_web_index"]=$settings["app_web_root"]."index.php";
  $settings["app_root_namespace"]="\\".$settings["app_directory_name"]."\\";
  $settings["app_templates_path"]=$settings["app_root_path"]."templates".DIRECTORY_SEPARATOR;
  $settings["app_resources_path"]=$settings["app_root_path"]."resources".DIRECTORY_SEPARATOR;
  $settings["app_forms_path"]=$settings["app_root_path"]."forms".DIRECTORY_SEPARATOR;
  $settings["gd_ttf"]=$settings["webdb_parent_path"]."fonts".DIRECTORY_SEPARATOR."arial.ttf";
  $settings["favicon_source"]="";
  $settings["webdb_web_root"]="/".$settings["webdb_directory_name"]."/";
  $settings["webdb_web_resources"]=$settings["webdb_web_root"]."resources/";
  $settings["webdb_web_index"]=$settings["webdb_web_root"]."index.php";
  $settings["nn_data_path"]=$settings["app_root_path"]."nn_data".DIRECTORY_SEPARATOR;
  $settings["nn_learn_rate"]=0.005;
  $settings["nn_network_filename"]=$settings["nn_data_path"]."network.txt";
  $settings["nn_images_filename"]=$settings["nn_data_path"]."t10k-images-14x14.idx3-ubyte";
  $settings["nn_labels_filename"]=$settings["nn_data_path"]."t10k-labels.idx1-ubyte";
}

#####################################################################################################

function home_page($form_config)
{
  global $settings;
  $page_params=array();

  $chart_data=\webdb\chart\initilize_chart();
  $chart_data["h"]=300;
  $chart_data["x_min"]=-10.2;
  $chart_data["x_max"]=10.2;
  $chart_data["y_min"]=-0.1;
  $chart_data["y_max"]=1.1;
  $chart_data["grid_x"]=0.2;
  $chart_data["grid_y"]=0.1;
  $plot_data=array();
  for ($i=-100;$i<=100;$i++)
  {
    $x=$i/10;
    $y=\nn\sigmoid_function($x);
    $plot_data[]=array($x,$y);
  }
  $chart_data=\webdb\chart\assign_plot_data($chart_data,$plot_data,0,1,"red","",false,true);
  $page_params["sigmoid"]=\webdb\chart\output_chart($chart_data);

  # ~~~
  $card=array();
  $card["inputs"]=array(2,3);
  $network=\nn\network_initialize(2,2,1);
  $network["hidden_layer"][0]["weights"][0]=0;
  $network["hidden_layer"][0]["weights"][1]=1;
  $network["hidden_layer"][1]["weights"][0]=0;
  $network["hidden_layer"][1]["weights"][1]=1;
  $network["output_layer"][0]["weights"][0]=0;
  $network["output_layer"][0]["weights"][1]=1;
  \nn\network_feedforward($network,$card);
  var_dump($network);
  die;
  # ~~~

  /*$images=\webdb\utils\load_bytes_from_file($settings["nn_images_filename"]);
  $labels=\webdb\utils\load_bytes_from_file($settings["nn_labels_filename"]);
  $data=array();
  $data["training_set"]=array(); # the set we use to train (8000)
  $data["testing_set"]=array(); # the set we use to test (2000)
  for ($i=0;$i<10000;$i++)
  {
    if (($i%5)!=0) # 4 out of every 5 cards is assigned to the training set, with every 5th card assigned to the test set
    {
      $data["training_set"][]=\nn\load_card($images,$labels,$i);
    }
    else
    {
      $data["testing_set"][]=\nn\load_card($images,$labels,$i);
    }
  }

  $network=\nn\network_initialize(14*14,7*7,1*10);

  if (file_exists($settings["nn_network_filename"])==true)
  {
    \nn\network_load($network);
  }

  for ($i=1;$i<=5000;$i++)
  {
    $card_num=mt_rand(0,count($data["training_set"])-1);
    $card=$data["training_set"][$card_num];
    \nn\network_feedforward($network,$card);
    \nn\network_train($network,$card["outputs"]);
  }

  \nn\network_save($network);

  $correct=0;
  $n=count($data["testing_set"]);
  for ($i=0;$i<$n;$i++)
  {
    $card=$data["testing_set"][$i];
    \nn\network_feedforward($network,$card);
    $guess=\nn\network_guess($network);
    if ($guess==$card["output"])
    {
      $correct++;
    }
  }
  $page_params["accuracy"]=sprintf("%.1f",$correct/$n*100);

  $card_num=mt_rand(0,$n-1);
  $card=$data["testing_set"][$card_num];
  \nn\network_feedforward($network,$card);
  $page_params["guess"]=\nn\network_guess($network);

  $resp_total=0;
  $n=count($network["output_layer"]);
  for ($i=0;$i<$n;$i++)
  {
    $resp_total+=$network["output_layer"][$i]["output"]+1;
  }
  $rows=array();
  for ($i=0;$i<$n;$i++)
  {
    $row_params=array();
    $row_params["output"]=sprintf("%.3f",($network["output_layer"][$i]["output"]+1)/$resp_total*100)."% [ ".$i." ]";
    $rows[]=\webdb\utils\template_fill("output_row",$row_params);
  }
  $page_params["output_rows"]=implode(PHP_EOL,$rows);

  $page_params["card_num"]=$card_num;
  $page_params["label"]=$data["testing_set"][$card_num]["output"];

  $page_params["input_layer"]=\nn\draw_neuron_grid($network,"input_layer",14,14,"output",0);
  $page_params["hidden_layer"]=\nn\draw_neuron_grid($network,"hidden_layer",7,7);
  $page_params["output_layer"]=\nn\draw_neuron_grid($network,"output_layer",1,10);

  $page_params["hidden_layer_map"]=\nn\draw_weight_map($network,"hidden_layer",7,7,14,14);
  $page_params["output_layer_map"]=\nn\draw_weight_map($network,"output_layer",1,10,7,7);*/

  $content=\webdb\utils\template_fill("home_page",$page_params);
  $title="nn";
  \webdb\utils\output_page($content,$title);
}

#####################################################################################################

function draw_neuron_grid(&$network,$layer,$x,$y,$key="output",$gap=50)
{
  $values=array_column($network[$layer],$key);
  $tile=300;
  $w=$x*($gap+$tile);
  $h=$y*($gap+$tile);
  $buffer=imagecreatetruecolor($w,$h);
  $color=imagecolorallocate($buffer,230,230,230);
  imagefill($buffer,0,0,$color);
  $n=count($values);
  for ($i=0;$i<$n;$i++)
  {
    $value=$values[$i];
    $value=\nn\neuron_scale_output($value);
    $color=imagecolorallocate($buffer,$value,$value,$value);
    $cx=intval(round(($i%$x+0.5)*($tile+$gap)));
    $cy=intval(round((floor($i/$x)+0.5)*($tile+$gap)));
    imagefilledellipse($buffer,$cx,$cy,$tile,$tile,$color);
    $color=imagecolorallocate($buffer,20,20,20);
    imageellipse($buffer,$cx,$cy,$tile,$tile,$color);
  }
  \webdb\graphics\scale_img($buffer,0.1,$w,$h);
  $output=\webdb\graphics\base64_image($buffer,"png");
  imagedestroy($buffer);
  return $output;
}

#####################################################################################################

function draw_weight_map(&$network,$layer,$x_layer,$y_layer,$x_sub,$y_sub)
{
  $cell=3;
  $gap=2;
  $tile_x=$x_sub*$cell;
  $tile_y=$y_sub*$cell;
  $w=$x_layer*($gap+$tile_x)+$gap;
  $h=$y_layer*($gap+$tile_y)+$gap;
  $buffer=imagecreatetruecolor($w,$h);
  $color=imagecolorallocate($buffer,210,210,255);
  imagefill($buffer,0,0,$color);

  $n=count($network[$layer]);
  for ($i=0;$i<$n;$i++)
  {
    $values=$network[$layer][$i]["weights"];

    $Lx=$i%$x_layer;
    $Ly=floor($i/$x_layer);

    $ox=$Lx*($gap+$tile_x)+$gap;
    $oy=$Ly*($gap+$tile_y)+$gap;

    $m=count($values);
    for ($j=0;$j<$m;$j++)
    {

      $Sx=$j%$x_sub;
      $Sy=floor($j/$x_sub);

      $x1=intval(floor($ox+$Sx*$cell));
      $y1=intval(floor($oy+$Sy*$cell));

      $x2=$x1+$cell-1;
      $y2=$y1+$cell-1;

      $value=\nn\neuron_scale_output($values[$j]);
      $value=max(0,$value);
      $value=min(255,$value);
      $color=imagecolorallocate($buffer,$value,$value,$value);
      imagefilledrectangle($buffer,$x1,$y1,$x2,$y2,$color);
    }
  }

  $output=\webdb\graphics\base64_image($buffer,"png");
  imagedestroy($buffer);
  return $output;
}

#####################################################################################################

function load_card(&$images,&$labels,$i)
{
  $card=array();
  \nn\load_card_image($card,$images,16+$i*196+1);
  \nn\load_card_label($card,$labels,8+$i+1);
  return $card;
}

#####################################################################################################

function load_card_image(&$card,&$images,$offset) // images is an array of 1,960,000 bytes, each one representing a pixel (0-255) of the 10,000 * 14x14 (196) images
{
  $inputs=array();
  for ($i=0;$i<196;$i++)
  {
    $input=$images[$i+$offset]; # 0..255
    $input=\nn\neuron_normalize_input($input); # -1..+1
    $inputs[]=$input;
  }
  $card["inputs"]=$inputs;
}

#####################################################################################################

function load_card_label(&$card,&$labels,$offset) # labels is an array of 10,000 bytes, each representing the answer of each image
{
  $output=$labels[$offset];
  $outputs=array();
  for ($i=0;$i<10;$i++)
  {
    if ($i==$output)
    {
      $outputs[]=1.0;
    }
    else
    {
      $outputs[]=-1.0;
    }
  }
  $card["outputs"]=$outputs;
  $card["output"]=$output;
}

#####################################################################################################

function network_initialize($inputs,$hidden,$outputs)
{
  $network=array();
  $previous=false;
  $network["input_layer"]=\nn\network_layer_initialize($inputs,$previous);
  $network["hidden_layer"]=\nn\network_layer_initialize($hidden,$network["input_layer"]);
  $network["output_layer"]=\nn\network_layer_initialize($outputs,$network["hidden_layer"]);
  return $network;
}

#####################################################################################################

function network_layer_initialize($size,&$previous_layer)
{
  $layer=array();
  for ($i=0;$i<$size;$i++)
  {
    $layer[]=\nn\neuron_initialize($previous_layer);
  }
  return $layer;
}

#####################################################################################################

function network_train(&$network,$outputs)
{
  $n=count($network["output_layer"]);
  for ($i=0;$i<$n;$i++)
  {
    $network["output_layer"][$i]["error"]=$outputs[$i]-$network["output_layer"][$i]["output"];
    \nn\neuron_train($network["output_layer"][$i]);
  }
  $n=count($network["hidden_layer"]);
  for ($i=0;$i<$n;$i++)
  {
    \nn\neuron_train($network["hidden_layer"][$i]);
  }
}

#####################################################################################################

function network_feedforward(&$network,$card)
{
  $n=count($card["inputs"]);
  for ($i=0;$i<$n;$i++)
  {
    $network["input_layer"][$i]["output"]=$card["inputs"][$i];
  }
  $n=count($network["hidden_layer"]);
  for ($i=0;$i<$n;$i++)
  {
    \nn\neuron_feedforward($network["hidden_layer"][$i]);
  }
  $n=count($network["output_layer"]);
  for ($i=0;$i<$n;$i++)
  {
    \nn\neuron_feedforward($network["output_layer"][$i]);
  }
}

#####################################################################################################

function network_guess(&$network)
{
  $best_output=-1;
  $guess=false;
  $n=count($network["output_layer"]);
  for ($i=0;$i<$n;$i++)
  {
    if ($network["output_layer"][$i]["output"]>$best_output)
    {
      $best_output=$network["output_layer"][$i]["output"];
      $guess=$i;
    }
  }
  return $guess;
}

#####################################################################################################

function network_save(&$network)
{
  global $settings;
  $save_data=array();
  $layer_weights=array();
  $n=count($network["hidden_layer"]);
  for ($i=0;$i<$n;$i++)
  {
    $neuron=$network["hidden_layer"][$i];
    $layer_weights[]=$neuron["weights"];
  }
  $save_data["hidden_layer_weights"]=$layer_weights;
  $layer_weights=array();
  $n=count($network["output_layer"]);
  for ($i=0;$i<$n;$i++)
  {
    $neuron=$network["output_layer"][$i];
    $layer_weights[]=$neuron["weights"];
  }
  $save_data["output_layer_weights"]=$layer_weights;
  file_put_contents($settings["nn_network_filename"],json_encode($save_data,JSON_PRETTY_PRINT));
}

#####################################################################################################

function network_load(&$network)
{
  global $settings;
  $save_data=file_get_contents($settings["nn_network_filename"]);
  $save_data=json_decode($save_data,true);
  $n=count($network["hidden_layer"]);
  for ($i=0;$i<$n;$i++)
  {
    $network["hidden_layer"][$i]["weights"]=$save_data["hidden_layer_weights"][$i];
  }
  $n=count($network["output_layer"]);
  for ($i=0;$i<$n;$i++)
  {
    $network["output_layer"][$i]["weights"]=$save_data["output_layer_weights"][$i];
  }
}

#####################################################################################################

function neuron_initialize(&$previous_layer)
{
  $neuron=array();
  $neuron["inputs"]=array();
  $neuron["weights"]=array();
  $n=0;
  if ($previous_layer!==false)
  {
    $n=count($previous_layer);
  }
  for ($i=0;$i<$n;$i++)
  {
    $neuron["inputs"][]=&$previous_layer[$i];
    $neuron["weights"][]=lcg_value();
  }
  $neuron["output"]=0.0;
  $neuron["bias"]=0.0;
  $neuron["error"]=0.0;
  return $neuron;
}

#####################################################################################################

function neuron_train(&$neuron)
{
  global $settings;
  #$delta=(1-$neuron["output"])*(1+$neuron["output"])*$neuron["error"]*$settings["nn_learn_rate"];
  $n=count($neuron["inputs"]);
  for ($i=0;$i<$n;$i++)
  {
    $neuron["inputs"][$i]["error"]+=$neuron["weights"][$i]*$neuron["error"];
    $neuron["weights"][$i]+=$neuron["inputs"][$i]["output"]*$delta;
  }
}

#####################################################################################################

function neuron_feedforward(&$neuron)
{
  $input_sum=0;
  $n=count($neuron["inputs"]);
  for ($i=0;$i<$n;$i++)
  {
    $input_sum+=$neuron["inputs"][$i]["output"]*$neuron["weights"][$i];
  }
  $neuron["output"]=\nn\sigmoid_function($input_sum+$neuron["bias"]);
  $neuron["error"]=0.0;
}

#####################################################################################################

function neuron_normalize_input($input) # 0..255
{
  $norm=$input/256;
  return $norm; # 0..1
}

#####################################################################################################

function neuron_scale_output($norm) # 0..1
{
  $output=$norm*256;
  return intval(floor($output)); # 0..255
}

#####################################################################################################

function sigmoid_function($x)
{
  return 1/(1+exp(-$x));
}

#####################################################################################################
