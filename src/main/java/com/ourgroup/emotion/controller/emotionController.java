package com.ourgroup.emotion.controller;

import javax.servlet.http.HttpServletRequest;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.client.RestTemplate;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;

import com.ourgroup.emotion.dto.SentenceDto;
import com.ourgroup.emotion.dto.resultDto;


@Controller
public class emotionController {
	
	@RequestMapping(value = "show", method = RequestMethod.POST)
	public String post(HttpServletRequest request, Model model) {
	    RestTemplate template = new RestTemplate();
		String url = "http://106.15.44.33:5000/predict";   //直接访问
		String sentence = request.getParameter("input");
		SentenceDto sentenceDto = new SentenceDto();
		sentenceDto.setSentence(sentence);

	    // 封装参数，千万不要替换为Map与HashMap，否则参数无法传递
	    MultiValueMap<String, Object> paramMap = new LinkedMultiValueMap<String, Object>();
	    paramMap.add("input", sentenceDto);

	    resultDto result = template.postForObject(url, paramMap,  resultDto.class);

        model.addAttribute("result", "“"+sentence+"”"+"这句话是："+result.toString());

//		JSONObject itemJSONObj = JSONObject.parseObject(JSON.toJSONString(paramMap));
//		System.out.println(itemJSONObj.getString("input"));
		return "css";
		
	}
	

	
	@RequestMapping(value = "show", method = RequestMethod.GET)
	public String show() {
		return "css";
	}
}
