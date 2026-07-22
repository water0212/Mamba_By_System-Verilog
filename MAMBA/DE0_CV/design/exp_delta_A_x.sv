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
		
		input  logic is_first, // 保留不影響外層

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
	 * Q8 × Q16 = Q24
	 */
	assign mul_result = exp_deltaA_signed * $signed(x);

	// 修改：使用標準硬體四捨五入 (+32768 後右移)，並用有號數變數接住
	logic signed [MUL_SIZE-1:0] mul_rounded;
	logic signed [MUL_SIZE-1:0] mul_shifted;

	assign mul_rounded = mul_result + 65'sd32768;
	assign mul_shifted = mul_rounded >>> 16; // 絕對保證是算術右移

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