module Discretization
	#(
		parameter A_size = 16,
		parameter B_size = 16,
		parameter Delta_size = 16,
		parameter U_size = 16,
		parameter L = 4,
		parameter D_IN = 32,
		parameter N = 16
	)
	(	
		input  logic clk,
		input  logic rst,
		input  logic start,
		input  logic [15:0] data,   
		
		output logic out_valid,
		output logic [31:0] out_data, 
		
		output logic start_delta_mul,
		output logic finish
	);
	
	

	//////////////////////////////////////////////////////  INPUT
	
	logic start_a;
	logic [A_size-1:0] reg_A [0:D_IN-1][0:N-1];
	
	logic start_b;
	logic [B_size-1:0] reg_B [0:L-1][0:N-1];
	
	logic start_delta;
	logic [Delta_size-1:0] reg_delta [0:L-1][0:D_IN-1];
	
	logic start_u;
	logic signed [U_size	-1:0] reg_u [0:L-1][0:D_IN-1];
	
	logic data_cnt_rst;
	logic [0:10] data_cnt;
	
	typedef enum {IDLE, START, A, B, DELTA, U, START_MUL} input_state;
	input_state in_ps, in_ns;
	
	//**********************
	// COUNTER
	//**********************
	always_ff @(posedge clk) begin
		if (rst | data_cnt_rst) data_cnt <= 0;
		else data_cnt <= data_cnt + 1;
	end
	
	//**********************
	// A_input (d_in, n)
	//**********************
	always_ff @(posedge clk) begin
		if (start_a) reg_A[data_cnt / N][data_cnt % N] <= data;
	end
	
	//**********************
	// B_input (l, n)
	//**********************
	always_ff @(posedge clk) begin
		if (start_b) reg_B[data_cnt / N][data_cnt % N] <= data;
	end
	
	//*************************
	// delta_input (l, d_in)
	//*************************
	always_ff @(posedge clk) begin
		if (start_delta) begin
			reg_delta[data_cnt / D_IN][data_cnt % D_IN] <= data;
		end
	end
	
	//*************************
	// U_input (l, d_in)
	//*************************
	always_ff @(posedge clk) begin
		if (start_u) begin
			reg_u[data_cnt / D_IN][data_cnt % D_IN] <= data;
		end
	end
	
	//******************
	// IN_FSM
	//******************
	always_ff @(posedge clk) begin
		if (rst) in_ps <= IDLE;
		else in_ps <= in_ns;
	end
	
	always_comb begin
		data_cnt_rst    	= 0;
		start_a         	= 0;
		start_b         	= 0;
		start_delta     	= 0;
		start_u				= 0;
		start_delta_mul 	= 0;
		in_ns           	= in_ps;
		
		case (in_ps)
			IDLE: 
			begin
				in_ns = START;
			end
			START: 
			begin
				if (start) begin
					data_cnt_rst = 1;
					in_ns = A;
				end
			end
			A: 
			begin
				start_a = 1;
				if (data_cnt == D_IN*N-1) begin
					data_cnt_rst = 1;
					in_ns = B;
				end
			end
			B: 
			begin
				start_b = 1;
				if (data_cnt == L*N-1) begin
					data_cnt_rst = 1;
					in_ns = DELTA;
				end
			end
			DELTA: 
			begin
				start_delta = 1;
				if (data_cnt == L*D_IN-1) begin
					data_cnt_rst = 1;
					in_ns = U;
				end
			end
			U:
			begin
				start_u = 1;
				if (data_cnt == L*D_IN-1) begin
					data_cnt_rst = 1;
					in_ns = START_MUL;
				end
			end
			START_MUL: 
			begin
				start_delta_mul = 1;
				in_ns = IDLE;
			end
		endcase
	end

	////////////////////////////////////////////////////// MULTIPLIER
	
	
	localparam PE_NUM         	= 16;
	localparam ROW_GROUP      	= D_IN / PE_NUM;
	localparam DELTA_A_SIZE 	= Delta_size + A_size;
	localparam DELTA_B_SIZE 	= Delta_size + B_size;
	localparam DELTA_BU_SIZE	= 32;
	
	logic delta_mul_done; 
	logic delta_BU_finish;
	
	logic signed [DELTA_A_SIZE-1:0] delta_A [0:L-1][0:D_IN-1][0:N-1];
	logic signed [DELTA_B_SIZE-1:0] delta_B [0:L-1][0:D_IN-1][0:N-1];
	
	deltaA_deltaB 
	#(
		.A_size 			(A_size),
		.B_size			(B_size),
		.Delta_size 	(Delta_size),
		.L 				(L),
		.D_IN				(D_IN),
		.N					(N),
		.DELTA_A_SIZE	(DELTA_A_SIZE),
		.DELTA_B_SIZE	(DELTA_B_SIZE)
	)	delta_mul (	
		.clk			(clk),
		.rst			(rst),
		.start		(start_delta_mul),
		.reg_A 		(reg_A),
		.reg_B 		(reg_B),
		.reg_delta 	(reg_delta),
		
		.delta_A 	(delta_A),
		.delta_B 	(delta_B),
		.finish		(delta_mul_done)
	);
	
	logic signed [31:0] exp_in_data;
	logic        exp_start;
	logic [31:0] exp_out_data;
	logic        exp_finish;
	
	/*
		每個clk輸入一筆、輸出一筆
		clk1：in_data0, clk2：out_data0、in_data1
		
	*/
	
	Exponential #(.SIZE(32)) exp_unit (
		.clk			(clk),
		.rst			(rst),
		.in_valid	(exp_start),
		.data			(exp_in_data),
		.out_data	(exp_out_data),
		.out_valid	(exp_finish)
	);
	
	logic signed [DELTA_BU_SIZE-1:0] delta_BU [0:L-1][0:D_IN-1][0:N-1];
	
	deltaB_u
	#(
		.L 				(L),
		.D_IN 			(D_IN),
		.N					(N),
		.U_SIZE 			(U_size),
		.DELTA_B_SIZE	(DELTA_B_SIZE)
	) deltaBU_mul (
		.clk			(clk),
		.rst			(rst),
		.start		(delta_mul_done),
		.delta_B 	(delta_B),
		.u 			(reg_u),
		
		.data_out 	(delta_BU),
		.finish		(delta_BU_finish)
	);
	
	logic        exp_x_start;
	logic        exp_x_finish;
	logic signed [31:0] exp_x_out;

	
	/////////////////////////////////////////////////////// OUTPUT FSM
	
	logic out_busy;
	logic [0:$clog2(L)-1]    out_l;
	logic [0:$clog2(D_IN)-1] out_d;
	logic [0:$clog2(N)-1]    out_n;
	
	logic bu_done_flag; // 紀錄 deltaB_u 是否已經運算完畢的旗標
	
	typedef enum {INIT, WAIT_EXP, WAIT_EXP_X, WRITE} substate_t;
	substate_t sub_state;
	
	//保存上一個的 state：x[l-1][d][n]
	logic signed [31:0] state_x [0:D_IN-1][0:N-1];
	
	// 修改：使用標準硬體四捨五入 (+128 後右移)，並用有號數變數接住，防編譯器 Bug
	logic signed [31:0] bu_val;
	logic signed [31:0] bu_rounded;
	logic signed [31:0] bu_shifted;
	
	assign bu_val     = delta_BU[out_l][out_d][out_n];
	assign bu_rounded = bu_val + 32'sd128;
	assign bu_shifted = bu_rounded >>> 8; // 絕對保證是算術右移
	
	logic signed [31:0] current_x_new;
	assign current_x_new = exp_x_out + bu_shifted;
	
	//*****************
	//		x
	//*****************
	integer d_idx;
	integer n_idx;
	always_ff @(posedge clk) begin
		if (rst) begin
			for (d_idx = 0; d_idx < D_IN; d_idx = d_idx + 1) begin
				for (n_idx = 0; n_idx < N; n_idx = n_idx + 1) begin
					state_x[d_idx][n_idx] <= '0;
				end
			end
		end 
		// 對齊 Python 的 x = torch.zeros()：每次新運算開始時將狀態歸零
		else if (delta_mul_done) begin
			for (d_idx = 0; d_idx < D_IN; d_idx = d_idx + 1) begin
				for (n_idx = 0; n_idx < N; n_idx = n_idx + 1) begin
					state_x[d_idx][n_idx] <= '0;
				end
			end
		end
		// 在算出新的 x 時更新狀態
		else if (out_busy && sub_state == WAIT_EXP_X && exp_x_finish) begin
			state_x[out_d][out_n] <= current_x_new;
		end
	end
	
	
	exp_delta_A_x
	#(
		.EXP_SIZE      (32),
		.X_SIZE        (32),
		.OUT_SIZE      (32),
		.EXP_FRAC_BITS (8)
	) exp_deltaA_x_unit (
		.clk        (clk),
		.rst        (rst),
		.start      (exp_x_start),
		.is_first   (out_l == 0), // 判斷是否為 i==0

		.exp_deltaA (exp_out_data),
		.x          (state_x[out_d][out_n]),

		.data_out   (exp_x_out),
		.finish     (exp_x_finish)
	);
	
	
	
	
	
	always_ff @(posedge clk) begin
		if (rst) begin
			out_valid <= 0; out_data  <= 0; finish    <= 0;
			out_busy  <= 0; out_l <= 0; out_d <= 0; out_n <= 0; 
			exp_start <= 0; exp_x_start <= 0; sub_state <= INIT; exp_in_data <= 0;
			bu_done_flag <= 0;
		end 
		else begin
			// 隨時捕捉 delta_BU_finish 訊號
			if (delta_BU_finish) bu_done_flag <= 1;

			if (delta_mul_done) begin
				out_busy  <= 1; out_l <= 0; out_d <= 0; out_n <= 0; 
				out_valid <= 0; finish    <= 0; sub_state <= INIT; 
				exp_start <= 0; exp_x_start <= 0;
				bu_done_flag <= 0; // 新回合開始，重置旗標
			end 
			else if (out_busy) begin
				
				// 必須等待 BU 運算完成，才開始輸出最終的 X
				if (bu_done_flag || delta_BU_finish) begin
				
					// ========== delta_A 經過 Exponential 運算並乘上 X，再與 delta_BU 相加 ==========
					case (sub_state)
						INIT: begin
							exp_in_data <= delta_A[out_l][out_d][out_n];
							exp_start   <= 1;
							sub_state   <= WAIT_EXP;
						end
						
						WAIT_EXP: begin
							exp_start <= 0; 
							if (exp_finish) begin 
								exp_x_start <= 1; 
								sub_state   <= WAIT_EXP_X;
							end
						end
						
						WAIT_EXP_X: begin
							exp_x_start <= 0; 
							if (exp_x_finish) begin
								// 輸出相加後的 x = x_nonB + delta_BU
								out_data  <= current_x_new; 
								out_valid <= 1;
								sub_state <= WRITE;
							end
						end
						
						WRITE: begin
							out_valid <= 0;
							
							// 更新計數器 
							if (out_n == N-1) begin
								out_n <= 0;
								if (out_d == D_IN-1) begin
									out_d <= 0;
									if (out_l == L-1) begin
										out_l <= 0;
										out_busy <= 0;   
										finish   <= 1;   
										bu_done_flag <= 0; 
									end else out_l <= out_l + 1;
								end else out_d <= out_d + 1;
							end else out_n <= out_n + 1;
							
							sub_state <= INIT; 
						end
						
						default: sub_state <= INIT;
					endcase
				end
				else begin
					out_valid <= 0;
				end
				
			end 
			else begin
				out_valid <= 0; finish <= 0; 
			end
		end
	end

endmodule