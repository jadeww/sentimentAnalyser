package com.ourgroup.emotion.dto;

public class resultDto {

    private int result;

    public resultDto() {
    }

    public resultDto(int result) {
        this.result = result;
    }

    public int getResult() {
		return result;
	}

	public void setResult(int result) {
		this.result = result;
	}

	@Override
    public String toString() {
		if(result==1) {
			return "正评价";
		}
		else return "负评价";
    }
}
