module exp_delta_A_x
	#(
		parameter EXP_SIZE      = 32,
		parameter X_SIZE        = 32,
		parameter OUT_SIZE      = 32,
		parameter EXP_FRAC_BITS = 8
	)
	(
		input  logic clk,
		input  logic rst,
		input  logic start,
		
		input  logic is_first, // 判斷是否為 i==0

		// Exponential 模組輸出的 exp(delta_A)
		input  logic [EXP_SIZE-1:0] exp_deltaA,

		// 前一個時間點的 state x[l-1][d][n]
		input  logic signed [X_SIZE-1:0] x,

		// exp(delta_A) * x
		output logic signed [OUT_SIZE-1:0] data_out,

		output logic finish
	);

	
	/*************exp_deltaA 一定是正數。*************/
	/*******前面補 0，避免 MSB=1 時被當成負數。*********/
	
	logic signed [EXP_SIZE:0] exp_deltaA_signed;

	localparam MUL_SIZE = EXP_SIZE + 1 + X_SIZE;

	logic signed [MUL_SIZE-1:0] mul_result;
	logic signed [MUL_SIZE-1:0] scaled_result;

	assign exp_deltaA_signed = $signed({1'b0, exp_deltaA});

	/*
	 * 硬體的 mul_result = deltaA_int * x
	 * 相當於 Python 裡的 product / 256
	 */
	assign mul_result = exp_deltaA_signed * $signed(x);

	// 修改：數學化簡對齊 Python！只需加 128 並右移 8 bits
	logic signed [MUL_SIZE-1:0] mul_abs;
	logic signed [MUL_SIZE-1:0] mul_abs_rounded;
	logic signed [MUL_SIZE-1:0] mul_abs_shifted;
	logic signed [MUL_SIZE-1:0] mul_shifted;

	assign mul_abs         = (mul_result >= 0) ? mul_result : -mul_result;
	assign mul_abs_rounded = mul_abs + 65'sd128;                  // + 128
	assign mul_abs_shifted = mul_abs_rounded >>> EXP_FRAC_BITS;   // 右移 8 bits (除以 256)
	assign mul_shifted     = (mul_result >= 0) ? mul_abs_shifted : -mul_abs_shifted; 

	assign scaled_result = is_first ? mul_result : mul_shifted;

	always_ff @(posedge clk) begin
		if (rst) begin
			data_out <= 0;
			finish   <= 0;
		end else begin
			finish <= 0;

			if (start) begin
				data_out <= scaled_result;
				finish   <= 1;
			end
		end
	end
endmodule