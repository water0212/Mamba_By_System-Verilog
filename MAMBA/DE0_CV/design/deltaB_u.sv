module deltaB_u
	#(
		parameter L = 4,
		parameter D_IN = 16,
		parameter N = 16,
		parameter U_SIZE = 16,
		parameter DELTA_B_SIZE = 32,
		parameter DELTA_BU_SIZE = 32
	)
	(
		input logic clk,
		input logic rst,
		input logic start,
		input logic signed [DELTA_B_SIZE-1:0] delta_B [0:L-1][0:D_IN-1][0:N-1],
		input logic signed [U_SIZE-1:0] u [0:L-1][0:D_IN-1],
		
		output logic signed [DELTA_BU_SIZE-1:0] data_out [0:L-1][0:D_IN-1][0:N-1],
		output logic finish
	);
	
	localparam PE_NUM = 16;
	
	logic [$clog2(L)-1:0]l_cnt;
	logic [$clog2(D_IN)-1:0]d_cnt;
	logic [$clog2(N)-1:0]n_cnt;
	
	logic busy;
	integer p;
	
	always_ff @(posedge clk) begin
		if (rst) begin
			
			busy 		<= 0;
			finish 	<= 0;
		end else begin
		
			/************* PE_NUM個乘法器同時算 **************/
			
			for (p = 0; p < PE_NUM; p = p+1) begin
				if ((n_cnt+p) < N) begin
					data_out[l_cnt][d_cnt][n_cnt+p] <= $signed(delta_B[l_cnt][d_cnt][n_cnt+p]) * $signed(u[l_cnt][d_cnt]);
				end
			end
			
			if ((n_cnt+PE_NUM) >= N) begin	// 看這組是否完成
				n_cnt <= 0;
				
				if (d_cnt == D_IN-1) begin		// d 完成
					d_cnt <= 0;
					
					if (l_cnt == L-1) begin		// l 完成
						l_cnt 	<= 0;
						
						busy		<= 0;
						finish	<= 1;
					end else begin
						l_cnt <= l_cnt + 1;
					end
					
					
				end else begin
					d_cnt <= d_cnt + 1;
				end
				
			end else begin
				n_cnt <= n_cnt + PE_NUM;		// 下一組 n
			end
		end		
	end
	
endmodule
