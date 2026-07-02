module Exponential 
    #(
    parameter SIZE = 32
    ) 
    (
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic signed [SIZE-1:0] data,

    output logic [SIZE-1:0] out_data,
    output logic finish
    );
    
    
    //**********************
    //      data * 369
    //**********************
	 logic signed [SIZE+8:0] data_369;
    assign data_369 = $signed(data) * 369; // 強制有號數乘法

    //**********************
    //         >> 8
    //**********************
	 logic signed [SIZE:0]   shift_8_data;
    assign shift_8_data = data_369 >>> 8;  // 使用算術右移保持負號

    //**********************
    //          Z
    //**********************
	 logic signed [SIZE-1:0] z;
    assign z = shift_8_data >>> 8;         // 使用算術右移保持負號

    //**********************
    //          F
    //**********************
    logic [SIZE-1:0]        f;
    assign f = shift_8_data[7:0];

    //**********************
    //          2^f
    //**********************
	 logic [SIZE-1:0]        pow_2_f;
    assign pow_2_f = 256 + f;

    //**********************
    //          OUT
    //**********************
	 logic [SIZE-1:0]        shift_amount;
    logic [SIZE-1:0]        capped_shift; 
	 
    always_comb begin
        if (z < 0) begin
            shift_amount = -z; 
        end 
		  else begin
            shift_amount = z;
        end
        if (shift_amount > 31) begin
            capped_shift = 31;
        end 
		  else begin
            capped_shift = shift_amount;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            out_data <= 0;
            finish   <= 0;
        end 
		  else if (start) begin
            if (z < 0) begin
                out_data <= pow_2_f >> capped_shift;
            end 
				else begin
                out_data <= pow_2_f << capped_shift;
            end
            finish <= 1; // 運算完成，通知主程式
        end 
		  else begin
            finish <= 0; // 等待下一次 start
        end
    end

endmodule
