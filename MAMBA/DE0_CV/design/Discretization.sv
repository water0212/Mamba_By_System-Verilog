module Discretization
	#(
		parameter A_size = 16,
		parameter B_size = 16,
		parameter Delta_size = 16,
		parameter L = 4,
		parameter D_IN = 32,
		parameter N = 16
	)
	(	
		input logic clk,
		input logic rst,
		input logic start,
		input logic [0:15] data,
		
		// output等等寫
		output logic start_delta_mul,
		output logic finish
		
	);
	//////////////////////////////////////////////////////  INPUT
	
	// A
	logic start_a;
	logic [0:A_size-1] reg_A [0:D_IN-1][0:N-1];
	
	//	B
	logic start_b;
	logic [0:B_size-1] reg_B [0:L-1][0:N-1];
	
	//	DELTA
	logic start_delta;
	logic [0:Delta_size-1] reg_delta [0:D_IN-1][0:-1];
	
	// COUNTER
	logic data_cnt_rst;
	logic [0:10] data_cnt;
	
	// IN_FSM
	typedef enum {IDLE, START, A, B, DELTA, START_MUL} input_state;
	input_state in_ps, in_ns;
	
	
	
	//**********************
	//			COUNTER
	//**********************
	always_ff @(posedge clk) begin
		if (rst | data_cnt_rst) begin
			data_cnt <= 0;
		end else begin
			data_cnt <= data_cnt + 1;
		end
	end
	
	//**********************
	//			A_input  (d_in,n)
	//**********************
	always_ff @(posedge clk) begin
		if (start_a) begin
			reg_A[data_cnt%D_IN][data_cnt/N] <= data[0:A_size-1];
		end
	end
	
	//**********************
	//			B_input  (l,n)
	//**********************
	always_ff @(posedge clk) begin
		if (start_b) begin
			reg_B[data_cnt%L][data_cnt/L] <= data[0:B_size-1];
		end
	end
	
	//*************************
	//			delta_input  (d_in,l)
	//*************************
	always_ff @(posedge clk) begin
		if (start_delta) begin
			reg_delta[data_cnt/L][data_cnt%L] <= data[0:Delta_size-1];
		end
	end
	
	
	//******************
	//			IN_FSM
	//******************
	always_ff @(posedge clk) begin
		if (rst) begin
			in_ps <= IDLE;
		end else begin
			in_ps <= in_ns;
		end
	end
	
	always_comb begin
		
		data_cnt_rst		= 0;
		start_a				= 0;
		start_b				= 0;
		start_delta			= 0;
		start_delta_mul	= 0;
		in_ns 				= in_ps;
		
		case (in_ps)
			IDLE:	begin
				in_ns = START;
			end
			START: begin
				if (start) begin
					data_cnt_rst = 1;
					in_ns = A;
				end
			end
			A: begin
				start_a = 1;
				if (data_cnt >= D_IN*N) begin
					data_cnt_rst = 1;
					in_ns = B;
				end
			end
			B: begin
				start_b = 1;
				if (data_cnt >= L*N) begin
					data_cnt_rst = 1;
					in_ns = DELTA;
				end
			end
			DELTA: begin
				start_delta = 1;
				if (data_cnt >= L*D_IN) begin
					data_cnt_rst = 1;
					in_ns = START_MUL;
				end
				
			end
			START_MUL: begin
				start_delta_mul = 1;
				in_ns = IDLE;
			end
		endcase
		
	end
	//////////////////////////////////////////////////////	MULTIPLIER
	
	
	// parameter
	localparam PE_NUM 			= 16;
	localparam DELTA_A_SIZE 	= Delta_size + A_size;
	localparam DELTA_B_SIZE 	= Delta_size + B_size;
	localparam ROW_GROUP			= D_IN / PE_NUM;
	
	// counter
	logic [0:$clog2(L)-1] token_cnt;
	logic [0:$clog2(N)-1] col_cnt;
	logic [0:$clog2(ROW_GROUP)-1] row_group_cnt;
	logic token_cnt_rst;
	
	// PE_STATE
	
	logic delta_mul_busy;
	logic delta_mul_done;
	
	
	// delta_A & delta_B
	logic [0:DELTA_A_SIZE] delta_A [0:L-1][0:D_IN-1][0:N-1];		// delta_A (l,d_in,n)
	logic [0:DELTA_B_SIZE] delta_B [0:L-1][0:D_IN-1][0:N-1];		// delta_B (l,d_in,n)
	
	
	//********************************
	//		delta_A & delta_B control
	//********************************
	always_ff @(posedge clk) begin
		if (rst) begin
			token_cnt 		<= 0;
			col_cnt			<= 0;
			row_group_cnt	<= 0;
			delta_mul_busy	<= 0;
			delta_mul_done	<= 0;
			
			
		end else begin					// 再寫一個control unit  *******************************
			delta_mul_done <= 0;
			if (start_delta_mul) begin
				token_cnt		<= -1;
				delta_mul_busy <= 1;
			end else if (delta_mul_busy) begin
				if (row_group_cnt == ROW_GROUP-1) begin
					row_group_cnt <= 0;
					
					if (col_cnt == N-1) begin
						col_cnt <= 0;

						if (token_cnt == L-1) begin
							token_cnt      <= 0;
							delta_mul_busy <= 0;
							delta_mul_done <= 1;
						end else begin
							token_cnt <= token_cnt + 1;
						end

					end else begin
						token_cnt <= token_cnt + 1;
					end
				end else begin
					col_cnt <= col_cnt + 1;
				end
				
				
				
			end else begin
				row_group_cnt <= row_group_cnt + 1;
			end
			
			
		end
		
	end
	
	
	
	//************************
	//		delta_A & delta_B
	//************************
	
	generate
		genvar PE_cnt;
		for (PE_cnt=0;PE_cnt<PE_NUM;PE_cnt=PE_cnt+1) begin : delta_A_genereate
			
			always_ff @(posedge clk) begin
				if (delta_mul_busy) begin
					// delta_A
					delta_A[token_cnt][row_group_cnt*PE_NUM+PE_cnt][col_cnt] <= reg_delta[row_group_cnt*PE_NUM+PE_cnt][token_cnt] * reg_A[row_group_cnt*PE_NUM+PE_cnt][col_cnt];
					// delta_B
					delta_B[token_cnt][row_group_cnt*PE_NUM+PE_cnt][col_cnt] <= reg_delta[row_group_cnt*PE_NUM+PE_cnt][token_cnt] * reg_B[token_cnt][col_cnt];
				end
				
			end
		end
		
		
	endgenerate
	
	//////////////////////////////////////////////////////
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
endmodule
