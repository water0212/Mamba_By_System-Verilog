module deltaA_deltaB
	#(
		parameter A_size = 16,
		parameter B_size = 16,
		parameter Delta_size = 16,
		parameter L = 4,
		parameter D_IN = 32,
		parameter N = 16,
		parameter DELTA_A_SIZE = 32,
		parameter DELTA_B_SIZE = 32
	)
	(
		input logic clk,
		input logic rst,
		input logic start,
		input logic [A_size-1:0] reg_A [0:D_IN-1][0:N-1],
		input logic [B_size-1:0] reg_B [0:L-1][0:N-1],
		input logic [Delta_size-1:0] reg_delta [0:L-1][0:D_IN-1],
		
		output logic signed [DELTA_A_SIZE-1:0] delta_A [0:L-1][0:D_IN-1][0:N-1],
		output logic signed [DELTA_B_SIZE-1:0] delta_B [0:L-1][0:D_IN-1][0:N-1],
		output logic finish
	);
	
	localparam PE_NUM         = 16;
	localparam ROW_GROUP      = D_IN / PE_NUM;
	
	
	logic [0:$clog2(L)-1]         token_cnt;
	logic [0:$clog2(N)-1]         col_cnt;
	logic [0:$clog2(ROW_GROUP)-1] row_group_cnt;
	
	logic delta_mul_busy;
	
	
	always_ff @(posedge clk) begin
		if (rst) begin
			token_cnt      <= 0;
			col_cnt        <= 0;
			row_group_cnt  <= 0;
			delta_mul_busy <= 0;
			finish			<= 0;
		end 
		else begin					
			finish <= 0;
			if (start) begin
				token_cnt      <= 0; 
				col_cnt        <= 0;
				row_group_cnt  <= 0;
				delta_mul_busy <= 1;
			end 
			else if (delta_mul_busy) begin
				if (row_group_cnt == ROW_GROUP-1) begin
					row_group_cnt <= 0;
					if (col_cnt == N-1) begin
						col_cnt <= 0;
						if (token_cnt == L-1) begin
							token_cnt      <= 0;
							delta_mul_busy <= 0;
							finish			<= 1;
						end 
						else begin
							token_cnt <= token_cnt + 1;
						end
					end 
					else begin
						col_cnt <= col_cnt + 1;
					end
				end 
				else begin
					row_group_cnt <= row_group_cnt + 1;
				end
			end
		end
	end
	
	// 新增：用來接 generate 乘法結果的實體線路 (Wire)
	logic signed [DELTA_A_SIZE-1:0] mul_out_A [0:PE_NUM-1];
	logic signed [DELTA_B_SIZE-1:0] mul_out_B [0:PE_NUM-1];

	genvar PE_cnt;
	generate
		for (PE_cnt = 0; PE_cnt < PE_NUM; PE_cnt = PE_cnt + 1) begin : delta_A_generate
			// 乘法器改為連續賦值 (Combinational logic)，不涉及 Clock
			assign mul_out_A[PE_cnt] = $signed(reg_delta[token_cnt][row_group_cnt*PE_NUM+PE_cnt]) * $signed(reg_A[row_group_cnt*PE_NUM+PE_cnt][col_cnt]);
			assign mul_out_B[PE_cnt] = $signed(reg_delta[token_cnt][row_group_cnt*PE_NUM+PE_cnt]) * $signed(reg_B[token_cnt][col_cnt]);
		end
	endgenerate
	
	// 新增：將所有乘法結果在「同一個」 always_ff 區塊內寫入陣列，完美解決多重驅動
	integer i;
	always_ff @(posedge clk) begin
		if (delta_mul_busy) begin
			for (i = 0; i < PE_NUM; i = i + 1) begin
				// 完美對齊 einsum('l d, d n -> l d n')
				delta_A[token_cnt][row_group_cnt*PE_NUM+i][col_cnt] <= mul_out_A[i];
				
				// 完美對齊 einsum('l d, l n -> l d n')
				delta_B[token_cnt][row_group_cnt*PE_NUM+i][col_cnt] <= mul_out_B[i];
			end
		end
	end
	
endmodule